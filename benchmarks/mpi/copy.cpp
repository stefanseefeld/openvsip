//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for MPI point-to-point messaging. (aka "copy").

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/map.hpp>
#include <vsip_csl/profile.hpp>

#include <vsip_csl/c++0x.hpp>
#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;
using vsip_csl::is_same;
using vsip_csl::equal;
using vsip_csl::operator<<;



template <typename T,
	  typename SBlock,
	  typename DBlock,
	  typename ImplTag>
class Send;

template <int SendMode, int RecvMode> struct Impl_isend;
template <int SendMode, int RecvMode> struct Impl_persistent;



template <typename MapT>
inline index_type
map_processor_index(MapT const& map, processor_type pid)
{
  for (index_type i=0; i<map.processor_set().size(); ++i)
    if (map.processor_set().get(i) == pid)
      return i;
  return no_index;
}



/***********************************************************************
  Send class using MPI_isend.
***********************************************************************/

// mode | vector | struct
//   0  |     no |     no
//   1  |  early |     no
//   2  |  early |  early
//   3  |  early |   late
//   4  |   late |     no
//   5  |   late |   late

template <typename T,
	  typename SBlock,
	  typename DBlock,
	  int      SendMode,
	  int      RecvMode>
class Send<T, SBlock, DBlock, Impl_isend<SendMode, RecvMode> >
{
  typedef typename SBlock::map_type src_map_type;
  typedef typename DBlock::map_type dst_map_type;

  typedef typename impl::Distributed_local_block<DBlock>::type
		dst_local_block_t;
  typedef typename impl::Distributed_local_block<SBlock>::type
		src_local_block_t;

  // Constructor.
public:

  Send(
    Vector<T, SBlock> src,
    Vector<T, DBlock> dst)
  : src_    (src),
    dst_    (dst),
    src_data_(src_.local().block()),
    dst_data_(dst_.local().block())
  {
    src_map_type src_map_ = src_.block().map();
    dst_map_type dst_map_ = dst_.block().map();

    // Check assumptions ----------------------------------------------
    assert(src_.size() == dst_.size());
    
    // This benchmarking code assumes a particular data layout,
    // check that inputs match layout before proceeding.

    // Source is single subblock
    test_assert(src_map_.num_subblocks(0) == 1);
    assert(src_data_.stride(0) == 1);

    // Destination is single subblock
    test_assert(dst_map_.num_subblocks(0) == 1);
    assert(dst_data_.stride(0) == 1);

    processor_type proc = local_processor();

    // Setup source datatype. -----------------------------------------
    send_proc_ = *(src_map_.processor_begin(0));

    if (proc == send_proc_)
    {
      if (SendMode == 0)
      {
	send_datatype_ = MPI_FLOAT;
	send_length_   = src_.size();
      }
      if (SendMode == 1 || SendMode == 2 || SendMode == 3)
      {
	MPI_Type_vector(src_.size(),	// number of blocks
			1,		// elements per block
			1,		// stride between block starts
			MPI_FLOAT,
			&send0_datatype_);
	MPI_Type_commit(&send0_datatype_);

	send_datatype_ = send0_datatype_;
	send_length_   = 1;
      }
      if (SendMode == 2)
      {
	int	     lena[1]   = { 1 };
	MPI_Aint     loca[1]   = { 0 };
	MPI_Datatype typesa[1] = { send0_datatype_ };
	MPI_Type_struct(1, lena, loca, typesa, &send1_datatype_);
	MPI_Type_commit(&send1_datatype_);

	send_datatype_ = send1_datatype_;
	send_length_   = 1;
      }
    }


    // Setup destination datatype -------------------------------------
    recv_proc_ = *(dst_map_.processor_begin(0));
    if (proc == recv_proc_)
    {
      if (RecvMode == 0)
      {
	recv_datatype_ = MPI_FLOAT;
	recv_length_   = src_.size();
      }
      if (RecvMode == 1 || RecvMode == 2 || RecvMode == 3)
      {
	MPI_Type_vector(src_.size(),	// number of blocks
			1,		// elements per block
			1,		// stride between block starts
			MPI_FLOAT,
			&recv0_datatype_);
	MPI_Type_commit(&recv0_datatype_);

	recv_datatype_ = recv0_datatype_;
	recv_length_   = 1;
      }
      if (RecvMode == 2)
      {
	int	     lena[1]   = { 1 };
	MPI_Aint     loca[1]   = { 0 };
	MPI_Datatype typesa[1] = { recv0_datatype_ };
	MPI_Type_struct(1, lena, loca, typesa, &recv1_datatype_);
	MPI_Type_commit(&recv1_datatype_);

	recv_datatype_ = recv1_datatype_;
	recv_length_   = 1;
      }
    }
  }

  ~Send()
  {
    if (local_processor() == send_proc_)
    {
      if (SendMode == 1 || SendMode == 2 || SendMode == 3)
	MPI_Type_free(&send0_datatype_);
      if (SendMode == 2)
	MPI_Type_free(&send1_datatype_);
    }

    if (local_processor() == recv_proc_)
    {
      if (RecvMode == 1 || RecvMode == 2 || RecvMode == 3)
	MPI_Type_free(&recv0_datatype_);
      if (RecvMode == 2)
	MPI_Type_free(&recv1_datatype_);
    }
  }

  void operator()()
  {
    MPI_Request req;
    MPI_Status  status;
    int ierr;

    processor_type proc = local_processor();

    if (proc == send_proc_)
    {
      if (SendMode == 4 || SendMode == 5)
      {
	MPI_Type_vector(src_.size(),	// number of blocks
			1,		// elements per block
			1,		// stride between block starts
			MPI_FLOAT,
			&send0_datatype_);
	MPI_Type_commit(&send0_datatype_);

	send_datatype_ = send0_datatype_;
	send_length_   = 1;
      }
      if (SendMode == 3 || SendMode == 5)
      {
	int	     lena[1]   = { 1 };
	MPI_Aint     loca[1]   = { 0 };
	MPI_Datatype typesa[1] = { send0_datatype_ };
	MPI_Type_struct(1, lena, loca, typesa, &send1_datatype_);
	MPI_Type_commit(&send1_datatype_);

	send_datatype_ = send1_datatype_;
	send_length_   = 1;
      }

      ierr = MPI_Isend(const_cast<T*>(src_data_.ptr()),
		       send_length_, send_datatype_,
		       recv_proc_, 0, MPI_COMM_WORLD,
		       &req);
      assert(ierr == MPI_SUCCESS);
    }

    if (proc == recv_proc_)
    {
      if (RecvMode == 4 || RecvMode == 5)
      {
	MPI_Type_vector(dst_.size(),	// number of blocks
			1,		// elements per block
			1,		// stride between block starts
			MPI_FLOAT,
			&recv0_datatype_);
	MPI_Type_commit(&recv0_datatype_);

	recv_datatype_ = recv0_datatype_;
	recv_length_   = 1;
      }
      if (RecvMode == 3 || RecvMode == 5)
      {
	int	     lena[1]   = { 1 };
	MPI_Aint     loca[1]   = { 0 };
	MPI_Datatype typesa[1] = { recv0_datatype_ };
	MPI_Type_struct(1, lena, loca, typesa, &recv1_datatype_);
	MPI_Type_commit(&recv1_datatype_);

	recv_datatype_ = recv1_datatype_;
	recv_length_   = 1;
      }

      int ierr = MPI_Recv(dst_data_.ptr(),
			  recv_length_, recv_datatype_,
			  send_proc_, 0, MPI_COMM_WORLD, &status);
      assert(ierr == MPI_SUCCESS);
    }

    if (proc == send_proc_)
    {
      MPI_Status status;
      MPI_Wait(&req, &status);
    }

    if (proc == send_proc_)
    {
      if (SendMode == 4 || SendMode == 5)
	MPI_Type_free(&send0_datatype_);
      if (SendMode == 3 || SendMode == 5)
	MPI_Type_free(&send1_datatype_);
    }

    if (proc == recv_proc_)
    {
      if (RecvMode == 4 || RecvMode == 5)
	MPI_Type_free(&recv0_datatype_);
      if (RecvMode == 3 || RecvMode == 5)
	MPI_Type_free(&recv1_datatype_);
    }
  }

  // Member data.
private:
  Vector<T, SBlock>           src_;
  Vector<T, DBlock>           dst_;

  dda::Data<src_local_block_t, dda::in> src_data_;
  dda::Data<dst_local_block_t, dda::out> dst_data_;

  processor_type    send_proc_;
  int               send_length_;
  MPI_Datatype      send_datatype_;

  processor_type    recv_proc_;
  int               recv_length_;
  MPI_Datatype      recv_datatype_;

  MPI_Datatype      send0_datatype_;
  MPI_Datatype      send1_datatype_;
  MPI_Datatype      recv0_datatype_;
  MPI_Datatype      recv1_datatype_;
};



/***********************************************************************
  Send class using MPI_persistent.
***********************************************************************/

// mode | vector | struct
//   0  |     no |     no
//   1  |  early |     no
//   2  |  early |  early

template <typename T,
	  typename SBlock,
	  typename DBlock,
	  int      SendMode,
	  int      RecvMode>
class Send<T, SBlock, DBlock, Impl_persistent<SendMode, RecvMode> >
{
  typedef typename SBlock::map_type src_map_type;
  typedef typename DBlock::map_type dst_map_type;

  typedef typename impl::Distributed_local_block<DBlock>::type
		dst_local_block_t;
  typedef typename impl::Distributed_local_block<SBlock>::type
		src_local_block_t;

  // Constructor.
public:

  Send(
    Vector<T, SBlock> src,
    Vector<T, DBlock> dst)
  : src_    (src),
    dst_    (dst),
    src_data_(src_.local().block()),
    dst_data_(dst_.local().block())
  {
    src_map_type src_map_ = src_.block().map();
    dst_map_type dst_map_ = dst_.block().map();

    // Check assumptions ----------------------------------------------
    assert(src_.size() == dst_.size());
    
    // This benchmarking code assumes a particular data layout,
    // check that inputs match layout before proceeding.

    // Source is single subblock
    test_assert(src_map_.num_subblocks(0) == 1);
    assert(src_data_.stride(0) == 1);

    // Destination is single subblock
    test_assert(dst_map_.num_subblocks(0) == 1);
    assert(dst_data_.stride(0) == 1);

    processor_type proc = local_processor();

    // Setup source datatype. -----------------------------------------
    send_proc_ = *(src_map_.processor_begin(0));
    recv_proc_ = *(dst_map_.processor_begin(0));

    if (proc == send_proc_)
    {
      if (SendMode == 0)
      {
	send_datatype_ = MPI_FLOAT;
	send_length_   = src_.size();
      }
      if (SendMode == 1 || SendMode == 2)
      {
	MPI_Type_vector(src_.size(),	// number of blocks
			1,		// elements per block
			1,		// stride between block starts
			MPI_FLOAT,
			&send0_datatype_);
	MPI_Type_commit(&send0_datatype_);

	send_datatype_ = send0_datatype_;
	send_length_   = 1;
      }
      if (SendMode == 2)
      {
	int	     lena[1]   = { 1 };
	MPI_Aint     loca[1]   = { 0 };
	MPI_Datatype typesa[1] = { send0_datatype_ };
	MPI_Type_struct(1, lena, loca, typesa, &send1_datatype_);
	MPI_Type_commit(&send1_datatype_);

	send_datatype_ = send1_datatype_;
	send_length_   = 1;
      }

      
      MPI_Send_init(const_cast<T*>(src_data_.ptr()),
		    send_length_, send_datatype_,
		    recv_proc_, 0, MPI_COMM_WORLD,
		    &send_req_);
    }


    // Setup destination datatype -------------------------------------
    if (proc == recv_proc_)
    {
      if (RecvMode == 0)
      {
	recv_datatype_ = MPI_FLOAT;
	recv_length_   = src_.size();
      }
      if (RecvMode == 1 || RecvMode == 2)
      {
	MPI_Type_vector(src_.size(),	// number of blocks
			1,		// elements per block
			1,		// stride between block starts
			MPI_FLOAT,
			&recv0_datatype_);
	MPI_Type_commit(&recv0_datatype_);

	recv_datatype_ = recv0_datatype_;
	recv_length_   = 1;
      }
      if (RecvMode == 2)
      {
	int	     lena[1]   = { 1 };
	MPI_Aint     loca[1]   = { 0 };
	MPI_Datatype typesa[1] = { recv0_datatype_ };
	MPI_Type_struct(1, lena, loca, typesa, &recv1_datatype_);
	MPI_Type_commit(&recv1_datatype_);

	recv_datatype_ = recv1_datatype_;
	recv_length_   = 1;
      }

      int ierr = MPI_Recv_init(dst_data_.ptr(),
			       recv_length_, recv_datatype_, 
			       send_proc_, 0, MPI_COMM_WORLD,
			       &recv_req_);
      assert(ierr == MPI_SUCCESS);
    }
  }

  ~Send()
  {
    if (local_processor() == send_proc_)
    {
      if (SendMode == 1 || SendMode == 2)
	MPI_Type_free(&send0_datatype_);
      if (SendMode == 2)
	MPI_Type_free(&send1_datatype_);
      MPI_Request_free(&send_req_);
    }

    if (local_processor() == recv_proc_)
    {
      if (RecvMode == 1 || RecvMode == 2)
	MPI_Type_free(&recv0_datatype_);
      if (RecvMode == 2)
	MPI_Type_free(&recv1_datatype_);
      MPI_Request_free(&recv_req_);
    }
  }

  void operator()()
  {
    MPI_Status  status;
    int ierr;

    processor_type proc = local_processor();

    if (proc == send_proc_)
    {
      MPI_Start(&send_req_);
    }

    if (proc == recv_proc_)
    {
      MPI_Start(&recv_req_);
      MPI_Wait(&recv_req_, &status);
    }

    if (proc == send_proc_)
    {
      MPI_Wait(&send_req_, &status);
    }
  }

  // Member data.
private:
  Vector<T, SBlock>           src_;
  Vector<T, DBlock>           dst_;

  dda::Data<src_local_block_t, dda::in> src_data_;
  dda::Data<dst_local_block_t, dda::out> dst_data_;

  processor_type    send_proc_;
  int               send_length_;
  MPI_Datatype      send_datatype_;
  MPI_Request       send_req_;

  processor_type    recv_proc_;
  int               recv_length_;
  MPI_Datatype      recv_datatype_;
  MPI_Request       recv_req_;

  MPI_Datatype      send0_datatype_;
  MPI_Datatype      send1_datatype_;
  MPI_Datatype      recv0_datatype_;
  MPI_Datatype      recv1_datatype_;
};



/***********************************************************************
  t_copy: data copy benchmark wrapper
***********************************************************************/

template <typename T,
	  typename ImplTag>
struct t_copy : Benchmark_base
{
  char const* what() { return "t_copy"; }
  int ops_per_point(length_type)
  {
    return 1;
  } 

  int riob_per_point(length_type rows) {return ops_per_point(rows)*sizeof(T);}
  int wiob_per_point(length_type rows) {return ops_per_point(rows)*sizeof(T);}
  int mem_per_point(length_type  rows) {return ops_per_point(rows)*sizeof(T);}

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Dense<1, T, row1_type, Map<> > src_block_t;
    typedef Dense<1, T, row1_type, Map<> > dst_block_t;

    processor_type np   = vsip::num_processors();

    Vector<processor_type> pset0(1); pset0.put(0, 0);
    Vector<processor_type> pset1(1); pset1.put(0, np-1);

    Map<> map0(pset0, 1);
    Map<> map1(pset1, 1);

    Vector<float, src_block_t> src(size, map0);
    Vector<float, dst_block_t> dst(size, map1);

    Send<T, src_block_t, dst_block_t, ImplTag> send(src, dst);

    // Initialize src
    for (index_type i=0; i<src.local().size(); ++i)
    {
      index_type g_i = global_from_local_index(src, 0, i);
      src.local().put(i, T(g_i));
    }

    // Assign src to dst
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      send();
    t1.stop();

    // Check result.
    for (index_type i=0; i<dst.local().size(); ++i)
    {
      index_type g_i = global_from_local_index(dst, 0, i);
      test_assert(equal(dst.local().get(i), T(g_i)));
    }
    
    time = t1.delta();
  }

  t_copy() {}

  // Member data.
};



void
defaults(Loop1P& /*loop*/)
{
}



int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case 10: loop(t_copy<float, Impl_isend<0, 0> >()); break;
  case 11: loop(t_copy<float, Impl_isend<1, 1> >()); break;
  case 12: loop(t_copy<float, Impl_isend<2, 2> >()); break;
  case 13: loop(t_copy<float, Impl_isend<3, 3> >()); break;
  case 14: loop(t_copy<float, Impl_isend<4, 4> >()); break;
  case 15: loop(t_copy<float, Impl_isend<5, 5> >()); break;

  case 20: loop(t_copy<float, Impl_persistent<0, 0> >()); break;
  case 21: loop(t_copy<float, Impl_persistent<1, 1> >()); break;
  case 22: loop(t_copy<float, Impl_persistent<2, 2> >()); break;

  case  0:
    std::cout
      << "copy -- MPI point-to-point messaging (aka \"copy\").\n"
      << "                   vector  struct\n"
      << "                   ------  ------\n"
      << "  -10: isend           no      no\n"
      << "  -11: isend        early      no\n"
      << "  -12: isend        early   early\n"
      << "  -13: isend        early    late\n"
      << "  -14: isend         late      no\n"
      << "  -15: isend         late    late\n"
      << "  -20: persistent      no      no\n"
      << "  -21: persistent   early      no\n"
      << "  -22: persistent   early   early\n"
      ;
  default:
    return 0;
  }
  return 1;
}

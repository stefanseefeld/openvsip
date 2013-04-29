/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for MPI alltoall.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/map.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip/core/ops_info.hpp>

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

struct Impl_alltoall;
struct Impl_isend;
struct Impl_isend_x;
struct Impl_persistent;

template <bool CopyLocal>
struct Impl_alltoallv;

template <bool CopyLocal, bool OptSingleRow, int OptPhaseOrder>
struct Impl_persistent_x;

template <typename ImplTag>
struct Does_copy_local
{
  static bool const value = true;
};

template <bool CopyLocal,
	  bool OptSingleRow,
	  int  OptPhaseOrder>
struct Does_copy_local<
  Impl_persistent_x<CopyLocal, OptSingleRow, OptPhaseOrder> >
{
  static bool const value = CopyLocal;
};

template <bool CopyLocal>
struct Does_copy_local<Impl_alltoallv<CopyLocal> >
{
  static bool const value = CopyLocal;
};



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
  Send class using MPI_alltoall.
***********************************************************************/

template <typename T,
	  typename SBlock,
	  typename DBlock>
class Send<T, SBlock, DBlock, Impl_alltoall>
{
  typedef typename SBlock::map_type src_map_type;
  typedef typename DBlock::map_type dst_map_type;

  typedef typename get_block_layout<SBlock>::order_type src_order_type;
  typedef typename get_block_layout<DBlock>::order_type dst_order_type;

  typedef typename impl::Distributed_local_block<DBlock>::type
		dst_local_block_t;
  typedef typename impl::Distributed_local_block<SBlock>::type
		src_local_block_t;

  // Constructor.
public:

  Send(
    Matrix<T, SBlock> src,
    Matrix<T, DBlock> dst)
  : src_    (src),
    dst_    (dst),
    src_data_(src_.local().block()),
    dst_data_(dst_.local().block())
  {
    src_map_type src_map_ = src_.block().map();
    dst_map_type dst_map_ = dst_.block().map();

    // Check assumptions ----------------------------------------------
    
    // This benchmarking code assumes a particular data layout,
    // check that inputs match layout before proceeding.

    // Source is row-major, block distributed by row.
    test_assert((is_same<row2_type, src_order_type>::value));
    test_assert(src_map_.num_subblocks(0) >= 1);
    test_assert(src_map_.num_subblocks(1) == 1);
    test_assert(src_map_.distribution(0) == block);
    test_assert(src_.size(0) % src_map_.num_subblocks(0) == 0);

    // Destination is row- or column-major, block distributed by column.
//    assert((is_same<col2_type, dst_order_type>::value));
    test_assert(dst_map_.num_subblocks(0) == 1);
    test_assert(dst_map_.num_subblocks(1) >= 1);
    test_assert(dst_map_.distribution(1) == block);
    test_assert(dst_.size(1) % dst_map_.num_subblocks(1) == 0);

    // Create send-side datatype.
    length_type nrows_per_send = src_.size(0) / src_map_.num_subblocks(0);
    length_type ncols_per_recv = dst_.size(1) / dst_map_.num_subblocks(1);

    // Check that dimensions are divisible by num subblocks.
    // (not very general, but this is a benchmark).
    test_assert(nrows_per_send * src_map_.num_subblocks(0) == src_.size(0));
    test_assert(ncols_per_recv * dst_map_.num_subblocks(1) == dst_.size(0));

//    printf("(%d): nrows_per_send = %d,  ncols_per_recv = %d\n",
//	   rank, nrows_per_send, ncols_per_recv);

    // Setup source datatype. -----------------------------------------

    assert(src_data_.stride(1) == 1);

    MPI_Type_vector(nrows_per_send,	// number of blocks
		    ncols_per_recv,	// elements per block
		    src_.size(1),	// stride between block starts
		    MPI_FLOAT,
		    &tmp0_datatype_);
    MPI_Type_commit(&tmp0_datatype_);

    int		 lena[2]   = { 1, 1 };
    MPI_Aint	 loca[2]   = { 0, ncols_per_recv * sizeof(T) };
    MPI_Datatype typesa[2] = { tmp0_datatype_, MPI_UB };
    MPI_Type_struct(2, lena, loca, typesa, &src_datatype_);
    MPI_Type_commit(&src_datatype_);



    // Setup destination datatype -------------------------------------

    if (is_same<dst_order_type, col2_type>::value)
    {
      test_assert(dst_data_.stride(0) == 1);

      MPI_Type_vector (ncols_per_recv,	// number of blocks
		       1,
		       dst_.size(0),
		       MPI_FLOAT,
		       &tmp1_datatype_);
      MPI_Type_commit(&tmp1_datatype_);

      int		 len[3]   = { 1, 1, 1 };
      MPI_Aint	 loc[3]   = { 0, 0, sizeof(T) };
      MPI_Datatype types[3] = { tmp1_datatype_, MPI_LB, MPI_UB };
      MPI_Type_struct(3, len, loc, types, &tmp2_datatype_);
      MPI_Type_commit(&tmp2_datatype_);
      
      
      MPI_Type_hvector(nrows_per_send, 1, sizeof(float), tmp2_datatype_,
		       &dst_datatype_);
      MPI_Type_commit(&dst_datatype_);
    }
    else
    {
      test_assert(dst_data_.stride(1) == 1);

      MPI_Type_vector(nrows_per_send,	// number of blocks
		      ncols_per_recv,	// elements per block
		      ncols_per_recv,   // stride between block starts
		      MPI_FLOAT,
		      &dst_datatype_);
      MPI_Type_commit(&dst_datatype_);
    }

  }

  ~Send()
  {
    MPI_Type_free(&tmp0_datatype_);
    if (is_same<dst_order_type, col2_type>::value)
    {
      MPI_Type_free(&tmp1_datatype_);
      MPI_Type_free(&tmp2_datatype_);
    }
    MPI_Type_free(&src_datatype_);
    MPI_Type_free(&dst_datatype_);
  }

  void operator()()
  {
    MPI_Alltoall(const_cast<T*>(src_data_.ptr()), 1, src_datatype_,
		 dst_data_.ptr(), 1, dst_datatype_,
		 MPI_COMM_WORLD);
  }

  void test_fixup() {}

  // Member data.
private:
  Matrix<T, SBlock>           src_;
  Matrix<T, DBlock>           dst_;

  dda::Data<src_local_block_t, dda::in> src_data_;
  dda::Data<dst_local_block_t, dda::out> dst_data_;

  MPI_Datatype      src_datatype_;
  MPI_Datatype      dst_datatype_;
  MPI_Datatype      tmp0_datatype_;
  MPI_Datatype      tmp1_datatype_;
  MPI_Datatype      tmp2_datatype_;
};



/***********************************************************************
  Send class using MPI_alltoallv.
***********************************************************************/

template <typename T,
	  typename SBlock,
	  typename DBlock,
	  bool     CopyLocal>
class Send<T, SBlock, DBlock, Impl_alltoallv<CopyLocal> >
{
  typedef typename SBlock::map_type src_map_type;
  typedef typename DBlock::map_type dst_map_type;

  typedef typename get_block_layout<SBlock>::order_type src_order_type;
  typedef typename get_block_layout<DBlock>::order_type dst_order_type;

  typedef typename impl::Distributed_local_block<DBlock>::type
		dst_local_block_t;
  typedef typename impl::Distributed_local_block<SBlock>::type
		src_local_block_t;

  // Constructor.
public:

  Send(
    Matrix<T, SBlock> src,
    Matrix<T, DBlock> dst)
  : src_      (src),
    dst_      (dst),
    src_data_  (src_.local().block()),
    dst_data_  (dst_.local().block()),
    nproc_    (vsip::num_processors()),
    rank_     (vsip::local_processor()),
    src_cnts_ (new int[nproc_]),
    src_disps_(new int[nproc_]),
    dst_cnts_ (new int[nproc_]),
    dst_disps_(new int[nproc_])
  {
    src_map_type src_map_ = src_.block().map();
    dst_map_type dst_map_ = dst_.block().map();

    // Check assumptions ----------------------------------------------
    
    // This benchmarking code assumes a particular data layout,
    // check that inputs match layout before proceeding.

    // Source is row-major, block distributed by row.
    assert((is_same<row2_type, src_order_type>::value));
    assert(src_map_.num_subblocks(0) >= 1);
    assert(src_map_.num_subblocks(1) == 1);
    assert(src_map_.distribution(0) == block);
    assert(src_.size(0) % src_map_.num_subblocks(0) == 0);

    // Destination is row- or column-major, block distributed by column.
//    assert((is_same<col2_type, dst_order_type>::value));
    assert(dst_map_.num_subblocks(0) == 1);
    assert(dst_map_.num_subblocks(1) >= 1);
    assert(dst_map_.distribution(1) == block);
    assert(dst_.size(1) % dst_map_.num_subblocks(1) == 0);

    // Create send-side datatype.
    nrows_per_send_ = src_.size(0) / src_map_.num_subblocks(0);
    ncols_per_recv_ = dst_.size(1) / dst_map_.num_subblocks(1);

    // Check that dimensions are divisible by num subblocks.
    // (not very general, but this is a benchmark).
    assert(nrows_per_send_ * src_map_.num_subblocks(0) == src_.size(0));
    assert(ncols_per_recv_ * dst_map_.num_subblocks(1) == dst_.size(1));

//    printf("(%d): nrows_per_send = %d,  ncols_per_recv = %d\n",
//	   rank, nrows_per_send_, ncols_per_recv_);

    // Setup source datatype. -----------------------------------------

    assert(src_data_.stride(1) == 1);

    // MPI_alltoallv requires a valid src_datatype, even if local
    // processor is not sending anything.

    MPI_Type_vector(nrows_per_send_,	// number of blocks
		    ncols_per_recv_,	// elements per block
		    src_.size(1),	// stride between block starts
		    MPI_FLOAT,
		    &tmp0_datatype_);
    MPI_Type_commit(&tmp0_datatype_);
      
    int		 lena[2]   = { 1, 1 };
    MPI_Aint	 loca[2]   = { 0, ncols_per_recv_ * sizeof(T) };
    MPI_Datatype typesa[2] = { tmp0_datatype_, MPI_UB };
    MPI_Type_struct(2, lena, loca, typesa, &src_datatype_);
    MPI_Type_commit(&src_datatype_);

    for (index_type p=0; p<nproc_; ++p)
    {
      if (src_map_.subblock()  == no_subblock ||
	  dst_map_.subblock(p) == no_subblock ||
	  (!CopyLocal && p == rank_))
	src_cnts_[p]  = 0;
      else
      {
	src_cnts_[p]  = 1;
	src_disps_[p] = dst_map_.subblock(p);
      }
    }


    // Setup destination datatype -------------------------------------

    // MPI_alltoallv requires a valid dst_datatype, even if local
    // processor is not receiving anything.

    if (is_same<dst_order_type, col2_type>::value)
    {
      assert(dst_data_.stride(0) == 1);

      MPI_Type_vector (ncols_per_recv_,	// number of blocks
		       1,
		       dst_.size(0),
		       MPI_FLOAT,
		       &tmp1_datatype_);
      MPI_Type_commit(&tmp1_datatype_);

      int		 len[3]   = { 1, 1, 1 };
      MPI_Aint	 loc[3]   = { 0, 0, sizeof(T) };
      MPI_Datatype types[3] = { tmp1_datatype_, MPI_LB, MPI_UB };
      MPI_Type_struct(3, len, loc, types, &tmp2_datatype_);
      MPI_Type_commit(&tmp2_datatype_);
	
	
      MPI_Type_hvector(nrows_per_send_, 1, sizeof(float), tmp2_datatype_,
		       &dst_datatype_);
      MPI_Type_commit(&dst_datatype_);
    }
    else
    {
      assert(dst_data_.stride(1) == 1);
      
      MPI_Type_vector(nrows_per_send_,	// number of blocks
		      ncols_per_recv_,	// elements per block
		      ncols_per_recv_,   // stride between block starts
		      MPI_FLOAT,
		      &dst_datatype_);
      MPI_Type_commit(&dst_datatype_);
    }

    for (index_type p=0; p<nproc_; ++p)
    {
      if (dst_map_.subblock()  == no_subblock ||
	  src_map_.subblock(p) == no_subblock ||
	  (!CopyLocal && p == rank_))
	dst_cnts_[p]  = 0;
      else
      {
	dst_cnts_[p]  = 1;
	dst_disps_[p] = src_map_.subblock(p);
      }
    }

  }

  ~Send()
  {
    MPI_Type_free(&tmp0_datatype_);
    if (is_same<dst_order_type, col2_type>::value)
    {
      MPI_Type_free(&tmp1_datatype_);
      MPI_Type_free(&tmp2_datatype_);
    }
    MPI_Type_free(&src_datatype_);
    MPI_Type_free(&dst_datatype_);

    delete[] src_cnts_;
    delete[] src_disps_;
    delete[] dst_cnts_;
    delete[] dst_disps_;
  }

  void operator()()
  {
    MPI_Alltoallv(const_cast<T*>(src_data_.ptr()),
		  src_cnts_,
		  src_disps_,
		  src_datatype_,
		  dst_data_.ptr(), 
		  dst_cnts_,
		  dst_disps_,
		  dst_datatype_,
		  MPI_COMM_WORLD);
  }

  // Do things that we disabled for benchmarking but need to do for
  // correctness.
  void test_fixup()
  {
    index_type src_rank_ = map_processor_index(src_.block().map(), rank_);
    index_type dst_rank_ = map_processor_index(dst_.block().map(), rank_);

    if (!CopyLocal && src_rank_ != no_index && dst_rank_ != no_index)
    {
      if (is_same<dst_order_type, col2_type>::value)
      {
	T *dst = dst_data_.ptr() + dst_rank_*nrows_per_send_;
	T const *src = src_data_.ptr() + src_rank_*ncols_per_recv_;
	impl::transpose_unit(dst, src,
			     nrows_per_send_, ncols_per_recv_,
			     dst_data_.stride(1),
			     src_data_.stride(0));
      }
      else
      {
	T *dst = dst_data_.ptr() + dst_rank_*ncols_per_recv_*nrows_per_send_;
	T const *src = src_data_.ptr() + src_rank_*ncols_per_recv_;
	for (index_type i=0; i<nrows_per_send_; ++i)
	  memcpy(dst + i*ncols_per_recv_,
		 src + i*src_.size(1),
		 ncols_per_recv_ * sizeof(T));
      }
    }
  }

  // Member data.
private:
  Matrix<T, SBlock>           src_;
  Matrix<T, DBlock>           dst_;

  dda::Data<src_local_block_t, dda::in> src_data_;
  dda::Data<dst_local_block_t, dda::out> dst_data_;

  length_type       nproc_;
  processor_type    rank_;

  length_type       nrows_per_send_;
  length_type       ncols_per_recv_;

  int*              src_cnts_;
  int*              src_disps_;
  int*              dst_cnts_;
  int*              dst_disps_;

  MPI_Datatype      src_datatype_;
  MPI_Datatype      dst_datatype_;
  MPI_Datatype      tmp0_datatype_;
  MPI_Datatype      tmp1_datatype_;
  MPI_Datatype      tmp2_datatype_;
};



/***********************************************************************
  Send class using MPI_isend / MPI_recv
***********************************************************************/

template <typename T,
	  typename SBlock,
	  typename DBlock>
class Send<T, SBlock, DBlock, Impl_isend>
{
  typedef typename SBlock::map_type src_map_type;
  typedef typename DBlock::map_type dst_map_type;

  typedef typename get_block_layout<SBlock>::order_type src_order_type;
  typedef typename get_block_layout<DBlock>::order_type dst_order_type;

  typedef typename impl::Distributed_local_block<DBlock>::type
		dst_local_block_t;
  typedef typename impl::Distributed_local_block<SBlock>::type
		src_local_block_t;

  // Constructor.
public:

  Send(
    Matrix<T, SBlock> src,
    Matrix<T, DBlock> dst)
  : src_    (src),
    dst_    (dst),
    src_data_(src_.local().block()),
    dst_data_(dst_.local().block()),
    nproc_  (src_.block().map().num_subblocks(0)),
    req_    (new MPI_Request[nproc_])
  {
    src_map_type src_map_ = src_.block().map();
    dst_map_type dst_map_ = dst_.block().map();

    // Check assumptions ----------------------------------------------
    
    // This benchmarking code assumes a particular data layout,
    // check that inputs match layout before proceeding.

    // Source is row-major, block distributed by row.
    assert((is_same<row2_type, src_order_type>::value));
    assert(src_map_.num_subblocks(0) >= 1);
    assert(src_map_.num_subblocks(1) == 1);
    assert(src_map_.distribution(0) == block);
    assert(src_.size(0) % src_map_.num_subblocks(0) == 0);

    // Destination is row- or column-major, block distributed by column.
    assert(dst_map_.num_subblocks(0) == 1);
    assert(dst_map_.num_subblocks(1) >= 1);
    assert(dst_map_.distribution(1) == block);
    assert(dst_.size(1) % dst_map_.num_subblocks(1) == 0);

    // Number of senders == number of receivers
    assert(src_map_.num_subblocks(0) == dst_map_.num_subblocks(1));

    // Create send-side datatype.
    nrows_per_send_ = src_.size(0) / src_map_.num_subblocks(0);
    ncols_per_recv_ = dst_.size(1) / dst_map_.num_subblocks(1);

//    printf("(%d): nrows_per_send = %d,  ncols_per_recv = %d\n",
//	   rank, nrows_per_send_, ncols_per_recv_);

    // Setup source datatype. -----------------------------------------

    assert(src_data_.stride(1) == 1);

    MPI_Type_vector(nrows_per_send_,	// number of blocks
		    ncols_per_recv_,	// elements per block
		    src_.size(1),	// stride between block starts
		    MPI_FLOAT,
		    &tmp0_datatype_);
    MPI_Type_commit(&tmp0_datatype_);

    int		 lena[2]   = { 1, 1 };
    MPI_Aint	 loca[2]   = { 0, ncols_per_recv_ * sizeof(T) };
    MPI_Datatype typesa[2] = { tmp0_datatype_, MPI_UB };
    MPI_Type_struct(2, lena, loca, typesa, &src_datatype_);
    MPI_Type_commit(&src_datatype_);



    // Setup destination datatype -------------------------------------

    if (is_same<dst_order_type, col2_type>::value)
    {
      assert(dst_data_.stride(0) == 1);

      MPI_Type_vector (ncols_per_recv_,	// number of blocks
		       1,
		       dst_.size(0),
		       MPI_FLOAT,
		       &tmp1_datatype_);
      MPI_Type_commit(&tmp1_datatype_);

      int		 len[3]   = { 1, 1, 1 };
      MPI_Aint	 loc[3]   = { 0, 0, sizeof(T) };
      MPI_Datatype types[3] = { tmp1_datatype_, MPI_LB, MPI_UB };
      MPI_Type_struct(3, len, loc, types, &tmp2_datatype_);
      MPI_Type_commit(&tmp2_datatype_);
      
      
      MPI_Type_hvector(nrows_per_send_, 1, sizeof(float), tmp2_datatype_,
		       &dst_datatype_);
      MPI_Type_commit(&dst_datatype_);
    }
    else
    {
      assert(dst_data_.stride(1) == 1);

      MPI_Type_vector(nrows_per_send_,	// number of blocks
		      ncols_per_recv_,	// elements per block
		      ncols_per_recv_,   // stride between block starts
		      MPI_FLOAT,
		      &dst_datatype_);
      MPI_Type_commit(&dst_datatype_);
    }

  }

  ~Send()
  {
    MPI_Type_free(&tmp0_datatype_);
    if (is_same<dst_order_type, col2_type>::value)
    {
      MPI_Type_free(&tmp1_datatype_);
      MPI_Type_free(&tmp2_datatype_);
    }
    MPI_Type_free(&src_datatype_);
    MPI_Type_free(&dst_datatype_);

    delete[] req_;
  }

  void operator()()
  {
    for (index_type i=0; i<nproc_; ++i)
    {
      MPI_Isend(const_cast<T*>(src_data_.ptr()) + i*ncols_per_recv_,
		1, src_datatype_,
		i, 0, MPI_COMM_WORLD, &(req_[i]));
    }

    if (is_same<dst_order_type, col2_type>::value)
    {
      for (index_type i=0; i<nproc_; ++i)
      {
	MPI_Status status;
	int ierr = MPI_Recv(dst_data_.ptr() + i*nrows_per_send_,
			    1, dst_datatype_, i, 0,
			    MPI_COMM_WORLD, &status);
	assert(ierr == MPI_SUCCESS);
      }
    }
    else
    {
      for (index_type i=0; i<nproc_; ++i)
      {
	MPI_Status status;
	int ierr = MPI_Recv(dst_data_.ptr()
			       + i*nrows_per_send_*ncols_per_recv_,
			    1, dst_datatype_, i, 0,
			    MPI_COMM_WORLD, &status);
	assert(ierr == MPI_SUCCESS);
      }
    }

    for (index_type i=0; i<nproc_; ++i)
    {
      MPI_Status status;
      MPI_Wait(&(req_[i]), &status);
    }
  }

  void test_fixup() {}

  // Member data.
private:
  Matrix<T, SBlock>           src_;
  Matrix<T, DBlock>           dst_;

  dda::Data<src_local_block_t, dda::in> src_data_;
  dda::Data<dst_local_block_t, dda::out> dst_data_;

  length_type       nproc_;
  length_type       nrows_per_send_;
  length_type       ncols_per_recv_;

  MPI_Request*      req_;

  MPI_Datatype      src_datatype_;
  MPI_Datatype      dst_datatype_;
  MPI_Datatype      tmp0_datatype_;
  MPI_Datatype      tmp1_datatype_;
  MPI_Datatype      tmp2_datatype_;
};



/***********************************************************************
  Send class using MPI_isend / MPI_recv + local copy
***********************************************************************/

template <typename T,
	  typename SBlock,
	  typename DBlock>
class Send<T, SBlock, DBlock, Impl_isend_x>
{
  typedef typename SBlock::map_type src_map_type;
  typedef typename DBlock::map_type dst_map_type;

  typedef typename get_block_layout<SBlock>::order_type src_order_type;
  typedef typename get_block_layout<DBlock>::order_type dst_order_type;

  typedef typename impl::Distributed_local_block<DBlock>::type
		dst_local_block_t;
  typedef typename impl::Distributed_local_block<SBlock>::type
		src_local_block_t;

  // Constructor.
public:

  Send(
    Matrix<T, SBlock> src,
    Matrix<T, DBlock> dst)
  : src_    (src),
    dst_    (dst),
    src_data_(src_.local().block()),
    dst_data_(dst_.local().block()),
    nproc_  (src_.block().map().num_subblocks(0)),
    rank_   (local_processor()),
    req_    (new MPI_Request[nproc_])
  {
    src_map_type src_map_ = src_.block().map();
    dst_map_type dst_map_ = dst_.block().map();

    // Check assumptions ----------------------------------------------
    
    // This benchmarking code assumes a particular data layout,
    // check that inputs match layout before proceeding.

    // Source is row-major, block distributed by row.
    assert((is_same<row2_type, src_order_type>::value));
    assert(src_map_.num_subblocks(0) >= 1);
    assert(src_map_.num_subblocks(1) == 1);
    assert(src_map_.distribution(0) == block);
    assert(src_.size(0) % src_map_.num_subblocks(0) == 0);

    // Destination is row- or column-major, block distributed by column.
    assert(dst_map_.num_subblocks(0) == 1);
    assert(dst_map_.num_subblocks(1) >= 1);
    assert(dst_map_.distribution(1) == block);
    assert(dst_.size(1) % dst_map_.num_subblocks(1) == 0);

    // Number of senders == number of receivers
    assert(src_map_.num_subblocks(0) == dst_map_.num_subblocks(1));

    // Create send-side datatype.
    nrows_per_send_ = src_.size(0) / src_map_.num_subblocks(0);
    ncols_per_recv_ = dst_.size(1) / dst_map_.num_subblocks(1);

//    printf("(%d): nrows_per_send = %d,  ncols_per_recv = %d\n",
//	   rank, nrows_per_send_, ncols_per_recv_);

    // Setup source datatype. -----------------------------------------

    assert(src_data_.stride(1) == 1);

    MPI_Type_vector(nrows_per_send_,	// number of blocks
		    ncols_per_recv_,	// elements per block
		    src_.size(1),	// stride between block starts
		    MPI_FLOAT,
		    &tmp0_datatype_);
    MPI_Type_commit(&tmp0_datatype_);

    int		 lena[2]   = { 1, 1 };
    MPI_Aint	 loca[2]   = { 0, ncols_per_recv_ * sizeof(T) };
    MPI_Datatype typesa[2] = { tmp0_datatype_, MPI_UB };
    MPI_Type_struct(2, lena, loca, typesa, &src_datatype_);
    MPI_Type_commit(&src_datatype_);



    // Setup destination datatype -------------------------------------

    if (is_same<dst_order_type, col2_type>::value)
    {
      assert(dst_data_.stride(0) == 1);

      MPI_Type_vector (ncols_per_recv_,	// number of blocks
		       1,
		       dst_.size(0),
		       MPI_FLOAT,
		       &tmp1_datatype_);
      MPI_Type_commit(&tmp1_datatype_);

      int		 len[3]   = { 1, 1, 1 };
      MPI_Aint	 loc[3]   = { 0, 0, sizeof(T) };
      MPI_Datatype types[3] = { tmp1_datatype_, MPI_LB, MPI_UB };
      MPI_Type_struct(3, len, loc, types, &tmp2_datatype_);
      MPI_Type_commit(&tmp2_datatype_);
      
      
      MPI_Type_hvector(nrows_per_send_, 1, sizeof(float), tmp2_datatype_,
		       &dst_datatype_);
      MPI_Type_commit(&dst_datatype_);
    }
    else
    {
      assert(dst_data_.stride(1) == 1);

      MPI_Type_vector(nrows_per_send_,	// number of blocks
		      ncols_per_recv_,	// elements per block
		      ncols_per_recv_,   // stride between block starts
		      MPI_FLOAT,
		      &dst_datatype_);
      MPI_Type_commit(&dst_datatype_);
    }

  }

  ~Send()
  {
    MPI_Type_free(&tmp0_datatype_);
    if (is_same<dst_order_type, col2_type>::value)
    {
      MPI_Type_free(&tmp1_datatype_);
      MPI_Type_free(&tmp2_datatype_);
    }
    MPI_Type_free(&src_datatype_);
    MPI_Type_free(&dst_datatype_);

    delete[] req_;
  }

  void operator()()
  {
    // Send -----------------------------------------------------------
    for (index_type i=0; i<nproc_; ++i)
    {
      if (i != rank_)
	MPI_Isend(const_cast<T*>(src_data_.ptr()) + i*ncols_per_recv_,
		  1, src_datatype_,
		  i, 0, MPI_COMM_WORLD, &(req_[i]));
    }

    // Copy -----------------------------------------------------------
    if (is_same<dst_order_type, col2_type>::value)
    {
      T *dst = dst_data_.ptr() + rank_*nrows_per_send_;
      T const *src = src_data_.ptr() + rank_*ncols_per_recv_;
      impl::transpose_unit(dst, src,
			   nrows_per_send_, ncols_per_recv_,
			   dst_data_.stride(1),
			   src_data_.stride(0));
    }
    else
    {
      T *dst = dst_data_.ptr() + rank_*ncols_per_recv_*nrows_per_send_;
      T const *src = src_data_.ptr() + rank_*ncols_per_recv_;
      for (index_type i=0; i<nrows_per_send_; ++i)
	memcpy(dst + i*ncols_per_recv_,
	       src + i*src_.size(1),
	       ncols_per_recv_ * sizeof(T));
    }

    // Recv -----------------------------------------------------------
    if (is_same<dst_order_type, col2_type>::value)
    {
      for (index_type i=0; i<nproc_; ++i)
      {
	MPI_Status status;
	if (i != rank_)
	  MPI_Recv(dst_data_.ptr() + i*nrows_per_send_,
			      1, dst_datatype_, i, 0,
			      MPI_COMM_WORLD, &status);
      }
    }
    else
    {
      for (index_type i=0; i<nproc_; ++i)
      {
	MPI_Status status;
	if (i != rank_)
	  MPI_Recv(dst_data_.ptr()
			      + i*nrows_per_send_*ncols_per_recv_,
			      1, dst_datatype_, i, 0,
			      MPI_COMM_WORLD, &status);
      }
    }

    // Wait for sends to finish ---------------------------------------
    for (index_type i=0; i<nproc_; ++i)
    {
      MPI_Status status;
      if (i != rank_)
	MPI_Wait(&(req_[i]), &status);
    }
  }

  void test_fixup() {}

  // Member data.
private:
  Matrix<T, SBlock>           src_;
  Matrix<T, DBlock>           dst_;

  dda::Data<src_local_block_t, dda::in> src_data_;
  dda::Data<dst_local_block_t, dda::out> dst_data_;

  length_type       nproc_;
  index_type        rank_;
  length_type       nrows_per_send_;
  length_type       ncols_per_recv_;

  MPI_Request*      req_;

  MPI_Datatype      src_datatype_;
  MPI_Datatype      dst_datatype_;
  MPI_Datatype      tmp0_datatype_;
  MPI_Datatype      tmp1_datatype_;
  MPI_Datatype      tmp2_datatype_;
};



/***********************************************************************
  Send class using Persistent communications.
***********************************************************************/

template <typename T,
	  typename SBlock,
	  typename DBlock>
class Send<T, SBlock, DBlock, Impl_persistent>
{
  typedef typename SBlock::map_type src_map_type;
  typedef typename DBlock::map_type dst_map_type;

  typedef typename get_block_layout<SBlock>::order_type src_order_type;
  typedef typename get_block_layout<DBlock>::order_type dst_order_type;

  typedef typename impl::Distributed_local_block<DBlock>::type
		dst_local_block_t;
  typedef typename impl::Distributed_local_block<SBlock>::type
		src_local_block_t;

  // Constructor.
public:

  Send(
    Matrix<T, SBlock> src,
    Matrix<T, DBlock> dst)
  : src_    (src),
    dst_    (dst),
    src_data_(src_.local().block()),
    dst_data_(dst_.local().block()),
    nproc_  (src_.block().map().num_subblocks(0)),
    req_    (new MPI_Request[2*nproc_]),
    status_ (new MPI_Status[2*nproc_])
  {
    src_map_type src_map_ = src_.block().map();
    dst_map_type dst_map_ = dst_.block().map();

    // Check assumptions ----------------------------------------------
    
    // This benchmarking code assumes a particular data layout,
    // check that inputs match layout before proceeding.

    // Source is row-major, block distributed by row.
    assert((is_same<row2_type, src_order_type>::value));
    assert(src_map_.num_subblocks(0) >= 1);
    assert(src_map_.num_subblocks(1) == 1);
    assert(src_map_.distribution(0) == block);
    assert(src_.size(0) % src_map_.num_subblocks(0) == 0);

    // Destination is row- or column-major, block distributed by column.
    assert(dst_map_.num_subblocks(0) == 1);
    assert(dst_map_.num_subblocks(1) >= 1);
    assert(dst_map_.distribution(1) == block);
    assert(dst_.size(1) % dst_map_.num_subblocks(1) == 0);

    // Number of senders == number of receivers
    assert(src_map_.num_subblocks(0) == dst_map_.num_subblocks(1));

    // Create send-side datatype.
    nrows_per_send_ = src_.size(0) / src_map_.num_subblocks(0);
    ncols_per_recv_ = dst_.size(1) / dst_map_.num_subblocks(1);

//    printf("(%d): nrows_per_send = %d,  ncols_per_recv = %d\n",
//	   rank, nrows_per_send_, ncols_per_recv_);

    // Setup source datatype. -----------------------------------------

    assert(src_data_.stride(1) == 1);

    MPI_Type_vector(nrows_per_send_,	// number of blocks
		    ncols_per_recv_,	// elements per block
		    src_.size(1),	// stride between block starts
		    MPI_FLOAT,
		    &tmp0_datatype_);
    MPI_Type_commit(&tmp0_datatype_);

    int		 lena[2]   = { 1, 1 };
    MPI_Aint	 loca[2]   = { 0, ncols_per_recv_ * sizeof(T) };
    MPI_Datatype typesa[2] = { tmp0_datatype_, MPI_UB };
    MPI_Type_struct(2, lena, loca, typesa, &src_datatype_);
    MPI_Type_commit(&src_datatype_);

    for (index_type i=0; i<nproc_; ++i)
    {
      MPI_Send_init(const_cast<T*>(src_data_.ptr()) + i*ncols_per_recv_,
		    1, src_datatype_,
		    i, 0, MPI_COMM_WORLD, &(req_[i]));
    }



    // Setup destination datatype -------------------------------------

    if (is_same<dst_order_type, col2_type>::value)
    {
      assert(dst_data_.stride(0) == 1);

      MPI_Type_vector (ncols_per_recv_,	// number of blocks
		       1,
		       dst_.size(0),
		       MPI_FLOAT,
		       &tmp1_datatype_);
      MPI_Type_commit(&tmp1_datatype_);

      int		 len[3]   = { 1, 1, 1 };
      MPI_Aint	 loc[3]   = { 0, 0, sizeof(T) };
      MPI_Datatype types[3] = { tmp1_datatype_, MPI_LB, MPI_UB };
      MPI_Type_struct(3, len, loc, types, &tmp2_datatype_);
      MPI_Type_commit(&tmp2_datatype_);
      
      
      MPI_Type_hvector(nrows_per_send_, 1, sizeof(float), tmp2_datatype_,
		       &dst_datatype_);
      MPI_Type_commit(&dst_datatype_);

      for (index_type i=0; i<nproc_; ++i)
      {
	int ierr = MPI_Recv_init(dst_data_.ptr() + i*nrows_per_send_,
				 1, dst_datatype_, i, 0,
				 MPI_COMM_WORLD, &(req_[nproc_+i]));
	assert(ierr == MPI_SUCCESS);
      }
    }
    else
    {
      assert(dst_data_.stride(1) == 1);

      MPI_Type_vector(nrows_per_send_,	// number of blocks
		      ncols_per_recv_,	// elements per block
		      ncols_per_recv_,   // stride between block starts
		      MPI_FLOAT,
		      &dst_datatype_);
      MPI_Type_commit(&dst_datatype_);

      for (index_type i=0; i<nproc_; ++i)
      {
	int ierr = MPI_Recv_init(dst_data_.ptr()
				   + i*nrows_per_send_*ncols_per_recv_,
				 1, dst_datatype_, i, 0,
				 MPI_COMM_WORLD, &(req_[nproc_+i]));
	assert(ierr == MPI_SUCCESS);
      }
    }

  }

  ~Send()
  {
    MPI_Type_free(&tmp0_datatype_);
    if (is_same<dst_order_type, col2_type>::value)
    {
      MPI_Type_free(&tmp1_datatype_);
      MPI_Type_free(&tmp2_datatype_);
    }
    MPI_Type_free(&src_datatype_);
    MPI_Type_free(&dst_datatype_);

    for (index_type i=0; i<2*nproc_; ++i)
      MPI_Request_free(&(req_[i]));

    delete[] req_;
    delete[] status_;
  }

  void operator()()
  {
    MPI_Startall(2*nproc_, req_);
    MPI_Waitall (2*nproc_, req_, status_);
  }

  void test_fixup() {}

  // Member data.
private:
  Matrix<T, SBlock>           src_;
  Matrix<T, DBlock>           dst_;

  dda::Data<src_local_block_t, dda::in> src_data_;
  dda::Data<dst_local_block_t, dda::out> dst_data_;

  length_type       nproc_;
  length_type       nrows_per_send_;
  length_type       ncols_per_recv_;

  MPI_Request*      req_;
  MPI_Status*       status_;

  MPI_Datatype      src_datatype_;
  MPI_Datatype      dst_datatype_;
  MPI_Datatype      tmp0_datatype_;
  MPI_Datatype      tmp1_datatype_;
  MPI_Datatype      tmp2_datatype_;
};



/***********************************************************************
  Send class using Persistent communications + local copy.
***********************************************************************/


// Template Parameters
//   COPYLOCAL indicates whether local data should be copied.
//   OPTSINGLEROW if true, and if processors own a single row, will sendd
//     directly instead of using a derived datatype.
//   OPTPHASEORDER

template <typename T,
	  typename SBlock,
	  typename DBlock,
	  bool     CopyLocal,
	  bool     OptSingleRow,
	  int      OptPhaseOrder>
class Send<T, SBlock, DBlock,
	   Impl_persistent_x<CopyLocal, OptSingleRow, OptPhaseOrder> >
{
  typedef typename SBlock::map_type src_map_type;
  typedef typename DBlock::map_type dst_map_type;

  typedef typename get_block_layout<SBlock>::order_type src_order_type;
  typedef typename get_block_layout<DBlock>::order_type dst_order_type;

  typedef typename impl::Distributed_local_block<DBlock>::type
		dst_local_block_t;
  typedef typename impl::Distributed_local_block<SBlock>::type
		src_local_block_t;

  // Constructor.
public:

  Send(
    Matrix<T, SBlock> src,
    Matrix<T, DBlock> dst)
  : src_    (src),
    dst_    (dst),
    src_data_(src_.local().block()),
    dst_data_(dst_.local().block()),
    nproc_  (src_.block().map().num_subblocks(0)),
    rank_   (vsip::local_processor()),
    src_rank_   (no_index),
    dst_rank_   (no_index),
    src_req_    (new MPI_Request[dst_.block().map().num_subblocks(1)]),
    src_status_ (new MPI_Status [dst_.block().map().num_subblocks(1)]),
    dst_req_    (new MPI_Request[src_.block().map().num_subblocks(0)]),
    dst_status_ (new MPI_Status [src_.block().map().num_subblocks(0)]),
    n_src_req_  (0),
    n_dst_req_  (0)

  {
    src_map_type src_map_ = src_.block().map();
    dst_map_type dst_map_ = dst_.block().map();

    // Check assumptions ----------------------------------------------
    
    // This benchmarking code assumes a particular data layout,
    // check that inputs match layout before proceeding.

    // Source is row-major, block distributed by row.
    test_assert((is_same<row2_type, src_order_type>::value));
    test_assert(src_map_.num_subblocks(0) >= 1);
    test_assert(src_map_.num_subblocks(1) == 1);
    test_assert(src_map_.distribution(0) == block);
    test_assert(src_.size(0) % src_map_.num_subblocks(0) == 0);

    // Destination is row- or column-major, block distributed by column.
    test_assert(dst_map_.num_subblocks(0) == 1);
    test_assert(dst_map_.num_subblocks(1) >= 1);
    test_assert(dst_map_.distribution(1) == block);
    test_assert(dst_.size(1) % dst_map_.num_subblocks(1) == 0);

    // Number of senders == number of receivers
    test_assert(src_map_.num_subblocks(0) == dst_map_.num_subblocks(1));

    // Create send-side datatype.
    nrows_per_send_ = src_.size(0) / src_map_.num_subblocks(0);
    ncols_per_recv_ = dst_.size(1) / dst_map_.num_subblocks(1);

//    printf("(%d): nrows_per_send = %d,  ncols_per_recv = %d\n",
//	   rank, nrows_per_send_, ncols_per_recv_);

    length_type n_src_sb = src_map_.num_subblocks(0);
    length_type n_dst_sb = dst_map_.num_subblocks(1);

    // Setup source datatype. -----------------------------------------

    assert(src_data_.stride(1) == 1);

    // Find source rank.  This is the rank of local procesor in
    // the source map.  If local processor is not in map, the
    // rank is 'no_index'.

    // src_rank_ is initialized to 'no_index'
    for (index_type i=0; i<src_map_.processor_set().size(); ++i)
      if (src_map_.processor_set().get(i) == rank_)
      {
	src_rank_ = i;
	break;
      }

    if (src_rank_ != no_index)
    {
      int          count;
      MPI_Datatype datatype;

      if (OptSingleRow && nrows_per_send_ == 1)
      {
	// If sending a single row, send it directly instead of using
	// a MPI derived data type (in theory they should be the same
	// for this case ...)
	count    = ncols_per_recv_;
	datatype = MPI_FLOAT;
      }
      else
      {
	MPI_Type_vector(nrows_per_send_,	// number of blocks
			ncols_per_recv_,	// elements per block
			src_.size(1),	// stride between block starts
			MPI_FLOAT,
			&tmp0_datatype_);
	MPI_Type_commit(&tmp0_datatype_);
      
	int		 lena[2]   = { 1, 1 };
	MPI_Aint	 loca[2]   = { 0, ncols_per_recv_ * sizeof(T) };
	MPI_Datatype typesa[2] = { tmp0_datatype_, MPI_UB };
	MPI_Type_struct(2, lena, loca, typesa, &src_datatype_);
	MPI_Type_commit(&src_datatype_);

	count    = 1;
	datatype = src_datatype_;
      }

      for (index_type i=0; i<n_dst_sb; ++i)
      {
	index_type     dsb = OptPhaseOrder>0 ? (src_rank_+i) % n_dst_sb : i;
	processor_type dst = *(dst_map_.processor_begin(dsb));

	if (dst != rank_)
	  MPI_Send_init(const_cast<T*>(src_data_.ptr()) + dsb*ncols_per_recv_, // buf
			count,
			datatype,
			dst, 0, MPI_COMM_WORLD,
			&(src_req_[n_src_req_++]));
      }
    }


    // Setup destination datatype -------------------------------------

    // Find destination rank.  This is the rank of local procesor in
    // the destination map.  If local processor is not in map, the
    // rank is 'no_index'.

    // dst_rank_ is initialized to 'no_index'
    for (index_type i=0; i<dst_map_.processor_set().size(); ++i)
      if (dst_map_.processor_set().get(i) == rank_)
      {
	dst_rank_ = i;
	break;
      }

    if (dst_rank_ != no_index)
    {
      if (is_same<dst_order_type, col2_type>::value)
      {
	assert(dst_data_.stride(0) == 1);

	MPI_Type_vector (ncols_per_recv_,	// number of blocks
			 1,
			 dst_.size(0),
			 MPI_FLOAT,
			 &tmp1_datatype_);
	MPI_Type_commit(&tmp1_datatype_);
	
	int		 len[3]   = { 1, 1, 1 };
	MPI_Aint	 loc[3]   = { 0, 0, sizeof(T) };
	MPI_Datatype types[3] = { tmp1_datatype_, MPI_LB, MPI_UB };
	MPI_Type_struct(3, len, loc, types, &tmp2_datatype_);
	MPI_Type_commit(&tmp2_datatype_);
      
      
	MPI_Type_hvector(nrows_per_send_, 1, sizeof(float), tmp2_datatype_,
			 &dst_datatype_);
	MPI_Type_commit(&dst_datatype_);

	for (index_type i=0; i<n_src_sb; ++i)
	{
	  index_type     ssb = OptPhaseOrder>0 ? (dst_rank_- i) % n_src_sb : i;
	  processor_type src = *(src_map_.processor_begin(ssb));

	  if (src != rank_)
	    MPI_Recv_init(dst_data_.ptr() + ssb*nrows_per_send_,
			  1, dst_datatype_, src, 0,
			  MPI_COMM_WORLD,
			  &(dst_req_[n_dst_req_++]));
	}
      }
      else
      {
	assert(dst_data_.stride(1) == 1);

	MPI_Datatype datatype;
	int          count;

	if (OptSingleRow && nrows_per_send_ == 1)
	{
	  count    = ncols_per_recv_;
	  datatype = MPI_FLOAT;
	}
	else
	{
	  MPI_Type_vector(nrows_per_send_,	// number of blocks
			  ncols_per_recv_,	// elements per block
			  ncols_per_recv_,   // stride between block starts
			  MPI_FLOAT,
			  &dst_datatype_);
	  MPI_Type_commit(&dst_datatype_);

	  count    = 1;
	  datatype = dst_datatype_;
	}

	for (index_type i=0; i<n_src_sb; ++i)
	{
	  index_type     ssb = OptPhaseOrder>0 ? (dst_rank_-i) % n_src_sb : i;
	  processor_type src = *(src_map_.processor_begin(ssb));
	  if (src != rank_)
	    MPI_Recv_init(
			dst_data_.ptr() + ssb*nrows_per_send_*ncols_per_recv_,
			count,
			datatype,
			src, 0, MPI_COMM_WORLD,
			&(dst_req_[n_dst_req_++]));
	}
      }
    }

  }

  ~Send()
  {
    if (src_rank_ != no_index)
    {
      if (! (OptSingleRow && nrows_per_send_ == 1) )
      {
	MPI_Type_free(&tmp0_datatype_);
	MPI_Type_free(&src_datatype_);
      }
    }

    if (dst_rank_ != no_index)
    {
      if (is_same<dst_order_type, col2_type>::value)
      {
	MPI_Type_free(&tmp1_datatype_);
	MPI_Type_free(&tmp2_datatype_);
	MPI_Type_free(&dst_datatype_);
      }
      else if (! (OptSingleRow && nrows_per_send_ == 1) )
	MPI_Type_free(&dst_datatype_);
    }

    for (index_type i=0; i<n_src_req_; ++i)
      MPI_Request_free(&(src_req_[i]));
    for (index_type i=0; i<n_dst_req_; ++i)
      MPI_Request_free(&(dst_req_[i]));

    delete[] src_req_;
    delete[] dst_req_;
    delete[] src_status_;
    delete[] dst_status_;
  }

  void operator()()
  {
    if (OptPhaseOrder == 2)
    {
      for (index_type i=0; i<std::max(n_src_req_, n_dst_req_); ++i)
      {
	if (i < n_src_req_) MPI_Start(src_req_+i);
	if (i < n_dst_req_) MPI_Start(dst_req_+i);
	if (i < n_src_req_) MPI_Wait(src_req_+i,  src_status_ + i);
	if (i < n_dst_req_) MPI_Wait(dst_req_+i,  dst_status_ + i);
      }
      if (CopyLocal && src_rank_ != no_index && dst_rank_ != no_index)
      {
	if (is_same<dst_order_type, col2_type>::value)
	{
	  T *dst = dst_data_.ptr() + dst_rank_*nrows_per_send_;
	  T const *src = src_data_.ptr() + src_rank_*ncols_per_recv_;
	  impl::transpose_unit(dst, src,
			       nrows_per_send_, ncols_per_recv_,
			       dst_data_.stride(1),
			       src_data_.stride(0));
	}
	else
	{
	  T *dst = dst_data_.ptr() + dst_rank_*ncols_per_recv_*nrows_per_send_;
	  T const *src = src_data_.ptr() + src_rank_*ncols_per_recv_;
	  for (index_type i=0; i<nrows_per_send_; ++i)
	    memcpy(dst + i*ncols_per_recv_,
		   src + i*src_.size(1),
		   ncols_per_recv_ * sizeof(T));
	}
      }
    }
    else
    {
      if (src_req_ > 0)
	MPI_Startall(n_src_req_, src_req_);
      if (dst_req_ > 0)
	MPI_Startall(n_dst_req_, dst_req_);

      if (CopyLocal && src_rank_ != no_index && dst_rank_ != no_index)
      {
	if (is_same<dst_order_type, col2_type>::value)
	{
	  T *dst = dst_data_.ptr() + dst_rank_*nrows_per_send_;
	  T const *src = src_data_.ptr() + src_rank_*ncols_per_recv_;
	  impl::transpose_unit(dst, src,
			       nrows_per_send_, ncols_per_recv_,
			       dst_data_.stride(1),
			       src_data_.stride(0));
	}
	else
	{
	  T *dst = dst_data_.ptr() + dst_rank_*ncols_per_recv_*nrows_per_send_;
	  T const *src = src_data_.ptr() + src_rank_*ncols_per_recv_;
	  for (index_type i=0; i<nrows_per_send_; ++i)
	    memcpy(dst + i*ncols_per_recv_,
		   src + i*src_.size(1),
		   ncols_per_recv_ * sizeof(T));
	}
      }

      if (src_req_ > 0)
	MPI_Waitall(n_src_req_, src_req_, src_status_);
      if (dst_req_ > 0)
	MPI_Waitall(n_dst_req_, dst_req_, dst_status_);
    }
  }

  // Do things that we disabled for benchmarking but need to do for
  // correctness.
  void test_fixup()
  {
    if (!CopyLocal && src_rank_ != no_index && dst_rank_ != no_index)
    {
      if (is_same<dst_order_type, col2_type>::value)
      {
	T *dst = dst_data_.ptr() + dst_rank_*nrows_per_send_;
	T const *src = src_data_.ptr() + src_rank_*ncols_per_recv_;
	impl::transpose_unit(dst, src,
			     nrows_per_send_, ncols_per_recv_,
			     dst_data_.stride(1),
			     src_data_.stride(0));
      }
      else
      {
	T *dst = dst_data_.ptr() + dst_rank_*ncols_per_recv_*nrows_per_send_;
	T const *src = src_data_.ptr() + src_rank_*ncols_per_recv_;
	for (index_type i=0; i<nrows_per_send_; ++i)
	  memcpy(dst + i*ncols_per_recv_,
		 src + i*src_.size(1),
		 ncols_per_recv_ * sizeof(T));
      }
    }
  }

  // Member data.
private:
  Matrix<T, SBlock>           src_;
  Matrix<T, DBlock>           dst_;

  dda::Data<src_local_block_t, dda::in> src_data_;
  dda::Data<dst_local_block_t, dda::out> dst_data_;

  length_type       nproc_;
  index_type        rank_;
  index_type        src_rank_;
  index_type        dst_rank_;
  length_type       nrows_per_send_;
  length_type       ncols_per_recv_;

  MPI_Request*      src_req_;
  MPI_Status*       src_status_;
  MPI_Request*      dst_req_;
  MPI_Status*       dst_status_;

  length_type       n_src_req_;
  length_type       n_dst_req_;

  MPI_Datatype      src_datatype_;
  MPI_Datatype      dst_datatype_;
  MPI_Datatype      tmp0_datatype_;
  MPI_Datatype      tmp1_datatype_;
  MPI_Datatype      tmp2_datatype_;
};



/***********************************************************************
  t_alltoall
***********************************************************************/

template <typename T,
	  typename SrcOrder,
	  typename DstOrder,
	  typename ImplTag>
struct t_alltoall : Benchmark_base
{
  char const* what() { return "t_alltoall"; }
  int ops_per_point(length_type  rows)
  {
    if (Does_copy_local<ImplTag>::value)
      return rows*ratio_;
    else
    {
      processor_type np = vsip::num_processors();
      return ((np-1)*rows*ratio_)/np;
    }
  } 

  int riob_per_point(length_type rows) {return ops_per_point(rows)*sizeof(T);}
  int wiob_per_point(length_type rows) {return ops_per_point(rows)*sizeof(T);}
  int mem_per_point(length_type  rows) {return ops_per_point(rows)*sizeof(T);}

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Dense<2, T, SrcOrder, Map<> > src_block_t;
    typedef Dense<2, T, DstOrder, Map<> > dst_block_t;

    processor_type np   = vsip::num_processors();
    index_type     rank = vsip::local_processor();

    length_type nrows = size;
    length_type ncols = ratio_*size;

    Map<> root_map(1, 1);
    Map<> src_map(np, 1);
    Map<> dst_map(1, np);

    Matrix<float, src_block_t> chk(nrows, ncols, root_map);
    Matrix<float, src_block_t> src(nrows, ncols, src_map);
    Matrix<float, dst_block_t> dst(nrows, ncols, dst_map);

    Send<T, src_block_t, dst_block_t, ImplTag> send(src, dst);

    // Initialize chk
    if (root_map.subblock() != no_subblock)
    {
      for (index_type r=0; r<nrows; ++r)
	for (index_type c=0; c<ncols; ++c)
	  chk.local()(r, c) = T(rank*ncols*nrows+r*ncols+c);
    }

    // Scatter chk to src
    src = chk;

    // Clear chk
    chk = T();

    // Assign src to dst
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      send();
    t1.stop();

    send.test_fixup();
    
    // Gather dst to chk
    chk = dst;
    
    // Check chk.
    if (root_map.subblock() != no_subblock)
    {
      for (index_type r=0; r<nrows; ++r)
	for (index_type c=0; c<ncols; ++c)
	  test_assert(chk.local().get(r, c) == T(rank*ncols*nrows+r*ncols+c));
    }
    
    time = t1.delta();
  }

  t_alltoall(length_type ratio) : ratio_(ratio) {};

// Member data:
  length_type ratio_;
};



/***********************************************************************
  t_alltoall_fr -- fixed rows
***********************************************************************/

template <typename T,
	  typename SrcOrder,
	  typename DstOrder,
	  typename ImplTag>
struct t_alltoall_fr : Benchmark_base
{
  char const* what() { return "t_alltoall_fr"; }

  int ops_per_point(length_type /*size*/)
  {
    if (Does_copy_local<ImplTag>::value)
      return rows_;
    else
    {
      processor_type np = vsip::num_processors();
      return ((np-1)*rows_)/np;
    }
  } 

  int riob_per_point(length_type size) {return ops_per_point(size)*sizeof(T);}
  int wiob_per_point(length_type size) {return ops_per_point(size)*sizeof(T);}
  int mem_per_point(length_type  size) {return ops_per_point(size)*sizeof(T);}

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Dense<2, T, SrcOrder, Map<> > src_block_t;
    typedef Dense<2, T, DstOrder, Map<> > dst_block_t;

    // processor_type np   = vsip::num_processors();
    index_type     rank = vsip::local_processor();

    length_type nrows = rows_;
    length_type ncols = size;

    Map<> root_map(1, 1);
    Map<> src_map(src_pset_, src_pset_.size(), 1);
    Map<> dst_map(dst_pset_, 1, dst_pset_.size());

    Matrix<float, src_block_t> chk(nrows, ncols, root_map);
    Matrix<float, src_block_t> src(nrows, ncols, src_map);
    Matrix<float, dst_block_t> dst(nrows, ncols, dst_map);

    Send<T, src_block_t, dst_block_t, ImplTag> send(src, dst);

    // Initialize chk
    if (root_map.subblock() != no_subblock)
    {
      for (index_type r=0; r<nrows; ++r)
	for (index_type c=0; c<ncols; ++c)
	  chk.local()(r, c) = T(rank*ncols*nrows+r*ncols+c);
    }

    // Scatter chk to src
    src = chk;

    // Clear chk
    chk = T();

    // Assign src to dst
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      send();
    t1.stop();

    send.test_fixup();
    
    // Gather dst to chk
    chk = dst;
    
    // Check chk.
    if (root_map.subblock() != no_subblock)
    {
      for (index_type r=0; r<nrows; ++r)
	for (index_type c=0; c<ncols; ++c)
	{
	  if (!equal(chk.local().get(r, c), T(rank*ncols*nrows+r*ncols+c)))
	  {
	    std::cout << "error: " << r << ", " << c << ": "
		      << chk.local().get(r, c) << ", "
		      << T(rank*ncols*nrows+r*ncols+c)
		      << std::endl;
	  }
	  test_assert(equal(chk.local().get(r, c),
			    T(rank*ncols*nrows+r*ncols+c) ));
	}
    }
    
    time = t1.delta();
  }

  t_alltoall_fr(length_type rows)
    : rows_    (rows),
      src_pset_(vsip::processor_set()),
      dst_pset_(vsip::processor_set())
  {}

  t_alltoall_fr(length_type rows, length_type n1, length_type n2)
    : rows_    (rows),
      src_pset_(n1),
      dst_pset_(n2)
  {
    length_type            np   = vsip::num_processors();
    Vector<processor_type> pset = vsip::processor_set();

    for (index_type i=0; i<n1; ++i)
      src_pset_.put(i, pset.get(i));

    for (index_type i=0; i<n2; ++i)
      dst_pset_.put(i, pset.get(np-n2+i));
  }

// Member data:
  length_type            rows_;
  Vector<processor_type> src_pset_;
  Vector<processor_type> dst_pset_;
};



void
defaults(Loop1P& loop)
{
  loop.stop_       = static_cast<unsigned>(-1);
  loop.user_param_ = -1;
}



int
test(Loop1P& loop, int what)
{

  typedef row2_type Rt;
  typedef col2_type Ct;

  if (what < 100)
  {
    if (loop.user_param_ == -1)                  loop.user_param_ = 1;
    if (loop.stop_ == static_cast<unsigned>(-1)) loop.stop_       = 11;
  }
  else
  {
    if (loop.user_param_ == -1)                  loop.user_param_ = 48;
    if (loop.stop_ == static_cast<unsigned>(-1)) loop.stop_       = 15;
  }

  // Ratio is the number of columns per row.
  length_type ratio = loop.user_param_;
  length_type rows  = loop.user_param_;

  length_type np = vsip::num_processors();
  length_type n1 = (np > 1) ? np/2 : 1; 
  length_type n2 = (np > 1) ? np/2 : 1; 

  switch (what)
  {
  case  1: loop(t_alltoall<float, Rt, Rt, Impl_alltoall>(ratio)); break;
  case  2: loop(t_alltoall<float, Rt, Ct, Impl_alltoall>(ratio)); break;

  case 11: loop(t_alltoall<float, Rt, Rt, Impl_isend>(ratio)); break;
  case 12: loop(t_alltoall<float, Rt, Ct, Impl_isend>(ratio)); break;

  case 21: loop(t_alltoall<float, Rt, Rt, Impl_isend_x>(ratio)); break;
  case 22: loop(t_alltoall<float, Rt, Ct, Impl_isend_x>(ratio)); break;

  case 31: loop(t_alltoall<float, Rt, Rt, Impl_persistent>(ratio)); break;
  case 32: loop(t_alltoall<float, Rt, Ct, Impl_persistent>(ratio)); break;

  case 41: loop(t_alltoall<float, Rt, Rt, Impl_persistent_x<true, true, true> >(ratio)); break;
  case 42: loop(t_alltoall<float, Rt, Ct, Impl_persistent_x<true, true, true> >(ratio)); break;


  case 101: loop(t_alltoall_fr<float, Rt, Rt, Impl_alltoall>(rows)); break;
  case 102: loop(t_alltoall_fr<float, Rt, Ct, Impl_alltoall>(rows)); break;

  case 111: loop(t_alltoall_fr<float, Rt, Rt, Impl_isend>(rows)); break;
  case 112: loop(t_alltoall_fr<float, Rt, Ct, Impl_isend>(rows)); break;

  case 121: loop(t_alltoall_fr<float, Rt, Rt, Impl_isend_x>(rows)); break;
  case 122: loop(t_alltoall_fr<float, Rt, Ct, Impl_isend_x>(rows)); break;

  case 131: loop(t_alltoall_fr<float, Rt, Rt, Impl_persistent>(rows)); break;
  case 132: loop(t_alltoall_fr<float, Rt, Ct, Impl_persistent>(rows)); break;

  // Persistent - nocopy, single-row opt, phase = 2
  case 141: loop(t_alltoall_fr<float, Rt, Rt, Impl_persistent_x<false, true, 2> >(rows)); break;
  case 142: loop(t_alltoall_fr<float, Rt, Ct, Impl_persistent_x<false, true, 2> >(rows)); break;
  case 143: loop(t_alltoall_fr<float, Rt, Rt, Impl_persistent_x<false, true, 2> >(rows,n1,n2)); break;
  case 144: loop(t_alltoall_fr<float, Rt, Ct, Impl_persistent_x<false, true, 2> >(rows,n1,n2)); break;

  // Persistent - nocopy, single-row opt, phase = 1
  case 151: loop(t_alltoall_fr<float, Rt, Rt, Impl_persistent_x<false, true, 1> >(rows)); break;
  case 152: loop(t_alltoall_fr<float, Rt, Ct, Impl_persistent_x<false, true, 1> >(rows)); break;
  case 153: loop(t_alltoall_fr<float, Rt, Rt, Impl_persistent_x<false, true, 1> >(rows,n1,n2)); break;
  case 154: loop(t_alltoall_fr<float, Rt, Ct, Impl_persistent_x<false, true, 1> >(rows,n1,n2)); break;

  // Persistent - nocopy, single-row opt, phase = 0
  case 161: loop(t_alltoall_fr<float, Rt, Rt, Impl_persistent_x<false, true, 0> >(rows)); break;
  case 162: loop(t_alltoall_fr<float, Rt, Ct, Impl_persistent_x<false, true, 0> >(rows)); break;
  case 163: loop(t_alltoall_fr<float, Rt, Rt, Impl_persistent_x<false, true, 0> >(rows,n1,n2)); break;
  case 164: loop(t_alltoall_fr<float, Rt, Ct, Impl_persistent_x<false, true, 0> >(rows,n1,n2)); break;


  // Persistent - local copy, single-row opt, phase = 2
  case 201: loop(t_alltoall_fr<float, Rt, Rt, Impl_persistent_x<true, true, 2> >(rows)); break;
  case 202: loop(t_alltoall_fr<float, Rt, Ct, Impl_persistent_x<true, true, 2> >(rows)); break;

  // Alltoallv - local copy, single-row opt, phase = 2
  case 171: loop(t_alltoall_fr<float, Rt, Rt, Impl_alltoallv<false> >(rows)); break;
  case 172: loop(t_alltoall_fr<float, Rt, Ct, Impl_alltoallv<false> >(rows)); break;
  case 173: loop(t_alltoall_fr<float, Rt, Rt, Impl_alltoallv<false> >(rows,n1,n2)); break;
  case 174: loop(t_alltoall_fr<float, Rt, Ct, Impl_alltoallv<false> >(rows,n1,n2)); break;

  case 181: loop(t_alltoall_fr<float, Rt, Rt, Impl_alltoallv<true> >(rows)); break;
  case 182: loop(t_alltoall_fr<float, Rt, Ct, Impl_alltoallv<true> >(rows)); break;
  case 183: loop(t_alltoall_fr<float, Rt, Rt, Impl_alltoallv<true> >(rows,n1,n2)); break;
  case 184: loop(t_alltoall_fr<float, Rt, Ct, Impl_alltoallv<true> >(rows,n1,n2)); break;

  case 0:
    std::cout
      << "alltoall -- MPI alltoall.\n"
      << "\n"
      << "  Method\n"
      << "  ------------\n"
      << "  alltoall     -- MPI_alltoall\n"
      << "  alltoallv    -- MPI_alltoall with local copy\n"
      << "  isend        -- MPI_isend / MPI_recv\n"
      << "  isend_x      -- MPI_isend / MPI_recv with local copy\n"
      << "  persistent   -- MPI 'persistant' communication\n"
      << "  persistent_x -- MPI 'persistant' communication with local copy\n"
      << "\n"
      << "    -1: rows -> rows, alltoall\n"
      << "    -2: rows -> cols, alltoall\n"
      << "\n"
      << "   -11: rows -> rows, isend\n"
      << "   -12: rows -> cols, isend\n"
      << "\n"
      << "   -21: rows -> rows, isend_x\n"
      << "   -22: rows -> cols, isend_x\n"
      << "\n"
      << "   -31: rows -> rows, persistent\n"
      << "   -32: rows -> cols, persistent\n"
      << "\n"
      << "   -41: rows -> rows, persistent_x (copy, single row, phase=1)\n"
      << "   -42: rows -> cols, persistent_x (copy, single row, phase=1)\n"
      << "\n"
      << "  -101: rows -> rows, alltoall\n"
      << "  -102: rows -> cols, alltoall\n"
      << "\n"
      << "  -111: rows -> rows, isend\n"
      << "  -112: rows -> cols, isend\n"
      << "\n"
      << "  -121: rows -> rows, isend_x\n"
      << "  -122: rows -> cols, isend_x\n"
      << "\n"
      << "  -131: rows -> rows, persistent\n"
      << "  -132: rows -> cols, persistent\n"
      << "\n"
      << "--- FIXED ROWS from here down --- \n"
      << "\n"
      << "  -141: rows -> rows, persistent_x (nocopy, single row, phase=2)\n"
      << "  -142: rows -> cols, persistent_x (nocopy, single row, phase=2)\n"
      << "  -143: rows -> rows, persistent_x (nocopy, single row, phase=2) (np)\n"
      << "  -144: rows -> cols, persistent_x (nocopy, single row, phase=2) (np)\n"
      << "\n"
      << "  -151: rows -> rows, persistent_x (nocopy, single row, phase=1)\n"
      << "  -152: rows -> cols, persistent_x (nocopy, single row, phase=1)\n"
      << "  -153: rows -> rows, persistent_x (nocopy, single row, phase=1) (np)\n"
      << "  -154: rows -> cols, persistent_x (nocopy, single row, phase=1) (np)\n"
      << "\n"
      << "  -161: rows -> rows, persistent_x (nocopy, single row, phase=0)\n"
      << "  -162: rows -> cols, persistent_x (nocopy, single row, phase=0)\n"
      << "  -163: rows -> rows, persistent_x (nocopy, single row, phase=0) (np)\n"
      << "  -164: rows -> cols, persistent_x (nocopy, single row, phase=0) (np)\n"
      << "\n"
      << "  -201: rows -> rows, persistent_x (copy, single row, phase=2)\n"
      << "  -202: rows -> cols, persistent_x (copy, single row, phase=2)\n"
      << "\n"
      << "  -171: rows -> rows, alltoallv (nocopy)\n"
      << "  -172: rows -> cols, alltoallv (nocopy)\n"
      << "  -173: rows -> rows, alltoallv (nocopy) (np)\n"
      << "  -174: rows -> cols, alltoallv (nocopy) (np)\n"
      << "\n"
      << "  -181: rows -> rows, alltoallv (copy)\n"
      << "  -182: rows -> cols, alltoallv (copy)\n"
      << "  -183: rows -> rows, alltoallv (copy) (np)\n"
      << "  -184: rows -> cols, alltoallv (copy) (np)\n"
      << "\n"
      << "  Parameter:\n"
      << "               default\n"
      << "               ----------------\n"
      << "    -1..-99\n"
      << "      -param    1  Number of columns per row\n"
      << "      -stop    11  Stop at 2^11\n"
      << "    -100..\n"
      << "      -param   48  Number of rows\n"
      << "      -stop    15  Stop at 2^15\n"
      ;
  default:
    return 0;
  }
  return 1;
}

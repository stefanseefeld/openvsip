//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_PAR_ASSIGN_BLOCK_VECTOR_HPP
#define VSIP_CORE_PAR_ASSIGN_BLOCK_VECTOR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vector>
#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/parallel/assign.hpp>
#include <vsip/core/adjust_layout.hpp>

// Verbosity level:
//  0 - no debug info
//  1 - show functions called
//  2 - message size details
//  3 - data values

#define VSIP_IMPL_ABV_VERBOSE 0



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{



// Block-vector parallel assignment.
template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
class Par_assign<Dim, T1, T2, Block1, Block2, Blkvec_assign>
  : Compile_time_assert<is_split_block<Block1>::value ==
                        is_split_block<Block2>::value>
{
  static dimension_type const dim = Dim;

  // disable_copy should only be set to true for testing purposes.  It
  // disables direct copy of data when source and destination are on
  // the same processor, causing chains to be built on both sides.
  // This is helps cover chain-to-chain copies for par-services-none.
  static bool const disable_copy = false;

  typedef typename Distributed_local_block<Block1>::type dst_local_block;
  typedef typename Distributed_local_block<Block2>::type src_local_block;

  typedef typename view_of<dst_local_block>::type dst_lview_type;
  typedef typename view_of<src_local_block>::const_type src_lview_type;

  typedef typename get_block_layout<src_local_block>::type raw_src_lp;
  typedef typename get_block_layout<dst_local_block>::type raw_dst_lp;

  typedef typename adjust_layout_packing<dense, raw_src_lp>::type
		src_lp;
  typedef typename adjust_layout_packing<dense, raw_dst_lp>::type
		dst_lp;

  typedef impl::Persistent_data<src_local_block, src_lp> src_ext_type;
  typedef impl::Persistent_data<dst_local_block, dst_lp> dst_ext_type;

  typedef typename Block1::map_type dst_appmap_t;
  typedef typename Block2::map_type src_appmap_t;

  typedef typename get_block_layout<Block1>::order_type dst_order_t;

  typedef impl::Communicator::request_type request_type;
  typedef impl::Communicator::chain_type   chain_type;

  /// A Msg_record holds a piece of a data transfer that together
  /// describe a complete communication.
  ///
  /// Members:
  ///   PROC_ is the remote processor (to send to or receive from),
  ///   SUBBLOCK_ is the local subblock to,
  ///   DATA_ is the raw data pointer of the local subblock,
  ///   CHAIN_ is the DMA chain representing the data from subblock_
  ///      to send.
  ///
  /// Notes:
  ///   [1] CHAIN_ completely describes the data to send/receive,
  ///       but it is dependent on the distributed blocks storage
  ///       location remaining unchanged from when the list is built
  ///       to when it is executed.  SUBBLOCK_ and DATA_ are stored
  ///       to check consistentcy and potentially update CHAIN_ if
  ///       the storage location changes.

  struct Msg_record
  {
    Msg_record(processor_type proc, index_type sb, stride_type offset,
	       length_type size)
      : proc_    (proc),
        subblock_(sb),
	offset_  (offset),
	size_    (size)
      {}

  public:
    processor_type      proc_;    // destination processor
    index_type          subblock_;
    stride_type         offset_;
    length_type         size_;
  };



  /// A Copy_record holds part of a data transfer where the source
  /// and destination processors are the same.
  ///
  /// Members:
  ///   SRC_SB_ is the source local subblock,
  ///   DST_SB_ is the destination local subblock,
  ///   SRC_DOM_ is the local domain within the source subblock to transfer,
  ///   DST_DOM_ is the local domain within the destination subblock to
  ///      transfer.

  struct Copy_record
  {
    Copy_record(index_type src_sb, index_type dst_sb,
	       Domain<Dim> src_dom,
	       Domain<Dim> dst_dom)
      : src_sb_  (src_sb),
        dst_sb_  (dst_sb),
	src_dom_ (src_dom),
	dst_dom_ (dst_dom)
      {}

  public:
    index_type     src_sb_;    // destination processor
    index_type     dst_sb_;
    Domain<Dim>    src_dom_;
    Domain<Dim>    dst_dom_;
  };


  // Constructor.
public:
  Par_assign(typename view_of<Block1>::type       dst,
	     typename view_of<Block2>::const_type src)
    : dst_      (dst),
      src_      (src.block()),
      dst_am_   (dst_.block().map()),
      src_am_   (src_.block().map()),
      comm_     (dst_am_.impl_comm()),
      send_list (),
      recv_list (),
      copy_list (),
      req_list  (),
      msg_count (0),
      src_ext_  (src_.local().block(), dda::in),
      dst_ext_  (dst_.local().block(), dda::out)
  {
    profile::Scope<profile::par> scope("Par_assign<Blkvec_assign>-cons");
    assert(src_am_.impl_comm() == dst_am_.impl_comm());

    build_send_list();
    if (!disable_copy)
      build_copy_list();
    build_recv_list();
  }

  ~Par_assign()
  {
    // At destruction, the list of outstanding sends should be empty.
    // This would be non-empty if:
    //  - Par_assign did not to clear the lists after
    //    processing it (library design error), or
    //  - User executed send() without a corresponding wait().
    assert(req_list.size() == 0);
  }


  // Implementation functions.
private:

  void build_send_list();
  void build_recv_list();
  void build_copy_list();

  void exec_send_list();
  void exec_recv_list();
  void exec_copy_list();

  void wait_send_list();

  void cleanup() {}	// Cleanup send_list buffers.


  // Invoke the parallel assignment
public:
  void operator()()
  {
    if (send_list.size() > 0) exec_send_list();
    if (copy_list.size() > 0) exec_copy_list();
    if (recv_list.size() > 0) exec_recv_list();

    if (req_list.size() > 0)  wait_send_list();

    cleanup();
  }


  // Private member data.
private:
  typename view_of<Block1>::type       dst_;
  typename view_of<Block2>::const_type src_;

  dst_appmap_t const& dst_am_;
  src_appmap_t const& src_am_;
  impl::Communicator& comm_;

  std::vector<Msg_record>    send_list;
  std::vector<Msg_record>    recv_list;
  std::vector<Copy_record>   copy_list;

  std::vector<request_type> req_list;

  int                       msg_count;

  src_ext_type              src_ext_;
  dst_ext_type              dst_ext_;
};



/***********************************************************************
  Definitions
***********************************************************************/

// Overload set for send, abstracts handling of interleaved- and
// split- complex.

template <typename T>
void
send(
  impl::Communicator&                            comm,
  processor_type                                 proc,
  T*                                             data,
  length_type                                    size,
  std::vector<impl::Communicator::request_type>& req_list)
{
  impl::Communicator::request_type   req;
  comm.send(proc, data, size, req);
  req_list.push_back(req);
}



template <typename T>
void
send(
  impl::Communicator&                            comm,
  processor_type                                 proc,
  std::pair<T*, T*> const&                       data,
  length_type                                    size,
  std::vector<impl::Communicator::request_type>& req_list)
{
  impl::Communicator::request_type   req1;
  impl::Communicator::request_type   req2;
  comm.send(proc, data.first,  size, req1);
  comm.send(proc, data.second, size, req2);
  req_list.push_back(req1);
  req_list.push_back(req2);
}



// Overload set for recv, abstracts handling of interleaved- and
// split- complex.

template <typename T>
inline void
recv(
  impl::Communicator& comm,
  processor_type      proc,
  T*                  data,
  length_type         size)
{
  comm.recv(proc, data, size);
}



template <typename T>
inline void
recv(
  impl::Communicator&      comm,
  processor_type           proc,
  std::pair<T*, T*> const& data,
  length_type              size)
{
  comm.recv(proc, data.first, size);
  comm.recv(proc, data.second, size);
}



// Build the send_list, a list of processor-subblock-local_domain
// records.  This can be done in advance of the actual assignment.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Blkvec_assign>::build_send_list()
{
  profile::Scope<profile::par> scope("Par_assign<Blkvec_assign>-build_send_list");
  processor_type rank = local_processor();

  length_type dsize  = dst_am_.impl_working_size();
  // std::min(dst_am_.num_subblocks(), dst_am_.impl_pvec().size());

#if VSIP_IMPL_ABV_VERBOSE >= 1
    std::cout << "(" << rank << ") "
	      << "build_send_list(dsize: " << dsize
	      << ") -------------------------------------\n";
#endif

  index_type src_sb = src_am_.subblock(rank);

  // If multiple processors have the subblock, the first processor
  // is responsible for sending it.

  if (src_sb != no_subblock &&
      *(src_am_.processor_begin(src_sb)) == rank)
  {
    assert(num_patches(src_, src_sb) == 1);
    Domain<dim> src_dom  = global_domain(src_, src_sb, 0);
    Domain<dim> src_ldom = local_domain (src_, src_sb, 0);

    src_ext_.begin();

    // Iterate over all processors
    for (index_type pi=0; pi<dsize; ++pi)
    {
      processor_type proc = dst_am_.impl_proc_from_rank(pi);

      // Transfers that stay on this processor is handled by the copy_list.
      if (!disable_copy && proc == rank)
	continue;

      index_type dst_sb = dst_am_.subblock(proc);

      if (dst_sb != no_subblock)
      {
	// Check to see if destination processor already has block
	if (!disable_copy && processor_has_block(src_am_, proc, src_sb))
	  continue;

	assert(num_patches(dst_, dst_sb) == 1);
	Domain<dim> dst_dom  = global_domain(dst_, dst_sb, 0);
	Domain<dim> intr;

	if (intersect(src_dom, dst_dom, intr))
	{
	  Index<dim>  offset   = first(intr) - first(src_dom);
	  Domain<dim> send_dom = domain(first(src_ldom) + offset,
					extent(intr));

	  stride_type xoff   = send_dom.first()  * src_ext_.stride(0);
	  length_type length = send_dom.length();
	  // stride = send_dom.stride() * src_ext_.stride(0) == 1
	  assert(send_dom.stride() * src_ext_.stride(0) == 1);

	  send_list.push_back(Msg_record(proc, src_sb, xoff, length));

#if VSIP_IMPL_ABV_VERBOSE >= 2
	      std::cout << "(" << rank << ") send "
			<< rank << "/" << src_sb << "/" << 0
			<< " -> "
			<< proc << "/" << dst_sb << "/" << 0
			<< " src: " << src_dom
			<< " dst: " << dst_dom
			<< " intr: " << intr
			<< " send: " << send_dom
		// << " val: " << get(local_view, first(send_dom))
			<< std::endl;
#endif
	}
	profile::Scope<profile::par> 
          scope("Par_assign<Blkvec_assign>-build_send_list-d");
      }
    }
    src_ext_.end();
  }
}



// Build the recv_list, a list of processor-subblock-local_domain
// records.  This can be done in advance of the actual assignment.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Blkvec_assign>::build_recv_list()
{
  profile::Scope<profile::par> 
    scope("Par_assign<Blkvec_assign>-build_recv_list");
  processor_type rank = local_processor();

  length_type ssize  = src_am_.impl_working_size();

#if VSIP_IMPL_ABV_VERBOSE >= 1
    std::cout << "(" << rank << ") "
	      << "build_recv_list(ssize: " << ssize
	      << ") -------------------------------------\n";
#endif

  index_type dst_sb = dst_am_.subblock(rank);

  if (dst_sb != no_subblock)
  {
    assert(num_patches(dst_, dst_sb) == 1);
    Domain<dim> dst_dom  = global_domain(dst_, dst_sb, 0);
    Domain<dim> dst_ldom = local_domain (dst_, dst_sb, 0);

    dst_ext_.begin();
      
    // Iterate over all sending processors
    for (index_type pi=0; pi<ssize; ++pi)
    {
      // Rotate message order so processors don't all send to 0,
      // then 1, etc (Currently does not work, it needs to take into
      // account the number of subblocks).
      // processor_type proc = (src_am_.impl_proc_from_rank(pi) + rank) % size;
      processor_type proc = src_am_.impl_proc_from_rank(pi);

      // Transfers that stay on this processor is handled by the copy_list.
      if (!disable_copy && proc == rank)
	continue;
      
      index_type src_sb = src_am_.subblock(proc);

      // If multiple processors have the subblock, the first processor
      // is responsible for sending it to us.

      if (src_sb != no_subblock &&
	  *(src_am_.processor_begin(src_sb)) == proc)
      {
	// Check to see if destination processor already has block
	if (!disable_copy && processor_has_block(src_am_, rank, src_sb))
	  continue;

	assert(num_patches(src_, src_sb) == 1);

	Domain<dim> src_dom  = global_domain(src_, src_sb, 0);
	    
	Domain<dim> intr;
	    
	if (intersect(dst_dom, src_dom, intr))
	{
	  Index<dim>  offset   = first(intr) - first(dst_dom);
	  Domain<dim> recv_dom = domain(first(dst_ldom) + offset,
					extent(intr));

	  ptrdiff_t   xoff   = recv_dom.first()  * dst_ext_.stride(0);
	  // stride = recv_dom.stride() * dst_ext_.stride(0) == 1
	  assert(recv_dom.stride() * dst_ext_.stride(0) == 1);
	  length_type length = recv_dom.length();

	  recv_list.push_back(Msg_record(proc, dst_sb, xoff, length));
	      
#if VSIP_IMPL_ABV_VERBOSE >= 2
	      std::cout << "(" << rank << ") recv "
			<< rank << "/" << dst_sb << "/" << 0
			<< " <- "
			<< proc << "/" << src_sb << "/" << 0
			<< " dst: " << dst_dom
			<< " src: " << src_dom
			<< " intr: " << intr
			<< " recv: " << recv_dom
		// << " val: " << get(local_view, first(recv_dom))
			<< std::endl;
#endif
	}
      }
    }
    dst_ext_.end();
  }
}



template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Blkvec_assign>::build_copy_list()
{
  profile::Scope<profile::par>
    scope("Par_assign<Blkvec_assign>-build_copy_list");
  processor_type rank = local_processor();

#if VSIP_IMPL_ABV_VERBOSE >= 1
  std::cout << "(" << rank << ") "
	    << "build_copy_list(num_procs: " << src_am_.num_processors()
	    << ") -------------------------------------\n";
#endif

  index_type dst_sb = dst_am_.subblock(rank);
  if (dst_sb != no_subblock)
  {
    index_type src_sb = src_am_.subblock(rank);
    if (src_sb != no_subblock)
    {
      assert(num_patches(dst_, dst_sb) == 1);
      Domain<dim> dst_dom  = global_domain(dst_, dst_sb, 0);
      Domain<dim> dst_ldom = local_domain (dst_, dst_sb, 0);

      assert(num_patches(src_, src_sb) == 1);
      Domain<dim> src_dom  = global_domain(src_, src_sb, 0);
      Domain<dim> src_ldom = local_domain (src_, src_sb, 0);

      Domain<dim> intr;

      if (intersect(src_dom, dst_dom, intr))
      {
	Index<dim>  send_offset = first(intr) - first(src_dom);
	Domain<dim> send_dom    = domain(first(src_ldom) + send_offset,
					 extent(intr));
	Index<dim>  recv_offset = first(intr) - first(dst_dom);
	Domain<dim> recv_dom    = domain(first(dst_ldom) + recv_offset,
					 extent(intr));

	copy_list.push_back(Copy_record(src_sb, dst_sb, send_dom, recv_dom));

#if VSIP_IMPL_ABV_VERBOSE >= 2
	std::cout << "(" << rank << ")"
		  << "copy src: " << src_sb << "/" << sp
		  << " " << send_dom
		  << "  dst: " << dst_sb << "/" << dp
		  << " " << recv_dom
		  << std::endl;
#endif
      }
    }
  }
}


// Execute the send_list.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Blkvec_assign>::exec_send_list()
{
  profile::Scope<profile::par>
    scope("Par_assign<Blkvec_assign>-exec_send_list");

#if VSIP_IMPL_ABV_VERBOSE >= 1
  processor_type rank = local_processor();
  std::cout << "(" << rank << ") "
	    << "exec_send_list(size: " << send_list.size()
	    << ") -------------------------------------\n";
#endif
  typedef typename std::vector<Msg_record>::iterator sl_iterator;
  typedef typename src_ext_type::storage_type storage_type;

  sl_iterator sl_cur = send_list.begin();
  sl_iterator sl_end = send_list.end();
  for (; sl_cur != sl_end; ++sl_cur)
  {
    processor_type proc = (*sl_cur).proc_;

    src_ext_.begin();
    send(comm_, proc,
	 storage_type::offset(src_ext_.ptr(), (*sl_cur).offset_),
	 (*sl_cur).size_, req_list);
    src_ext_.end();
  }
}



// Execute the recv_list.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Blkvec_assign>::exec_recv_list()
{
  profile::Scope<profile::par>
    scope("Par_assign<Blkvec_assign>-exec_recv_list");

#if VSIP_IMPL_ABV_VERBOSE >= 1
  processor_type rank = local_processor();
  std::cout << "(" << rank << ") "
	    << "exec_recv_list(size: " << recv_list.size()
	    << ") -------------------------------------\n";
#endif

  typedef typename std::vector<Msg_record>::iterator rl_iterator;
  typedef typename dst_ext_type::storage_type storage_type;

  rl_iterator rl_cur = recv_list.begin();
  rl_iterator rl_end = recv_list.end();
    
  for (; rl_cur != rl_end; ++rl_cur)
  {
    processor_type proc = (*rl_cur).proc_;

    dst_ext_.begin();
    recv(comm_, proc,
	 storage_type::offset(dst_ext_.ptr(), (*rl_cur).offset_),
	 (*rl_cur).size_);
    dst_ext_.end();
  }
}



// Execute the copy_list.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Blkvec_assign>::exec_copy_list()
{
  profile::Scope<profile::par>
    scope("Par_assign<Blkvec_assign>-exec_copy_list");

#if VSIP_IMPL_ABV_VERBOSE >= 1
  processor_type rank = local_processor();
  std::cout << "(" << rank << ") "
	    << "exec_copy_list(size: " << copy_list.size()
	    << ") -------------------------------------\n";
#endif

  src_lview_type src_lview = get_local_view(src_);
  dst_lview_type dst_lview = get_local_view(dst_);

  typedef typename std::vector<Copy_record>::iterator cl_iterator;
  for (cl_iterator cl_cur = copy_list.begin();
       cl_cur != copy_list.end();
       ++cl_cur)
  {
    view_assert_local(src_, (*cl_cur).src_sb_);
    view_assert_local(dst_, (*cl_cur).dst_sb_);

    dst_lview((*cl_cur).dst_dom_) = src_lview((*cl_cur).src_dom_);
#if VSIP_IMPL_ABV_VERBOSE >= 2
    std::cout << "(" << rank << ") "
	      << "src subblock: " << (*cl_cur).src_sb_ << " -> "
	      << "dst subblock: " << (*cl_cur).dst_sb_ << std::endl
#if VSIP_IMPL_ABV_VERBOSE >= 3
	      << dst_lview((*cl_cur).dst_dom_)
#endif
      ;
#endif
  }
}



// Wait for the send_list instructions to be completed.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Blkvec_assign>::wait_send_list()
{
  typename std::vector<request_type>::iterator
		cur = req_list.begin(),
		end = req_list.end();
  for(; cur != end; ++cur)
  {
    comm_.wait(*cur);
  }
  req_list.clear();
}


} // namespace vsip::impl

} // namespace vsip

#undef VSIP_IMPL_ABV_VERBOSE

#endif // VSIP_CORE_PARALLEL_ASSIGN_BLOCK_VECTOR_HPP

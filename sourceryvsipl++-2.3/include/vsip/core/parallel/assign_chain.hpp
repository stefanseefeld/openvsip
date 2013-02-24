// Copyright (c) 2005 - 2010 by CodeSourcery.  All rights reserved.

/// Description
///  Chained parallel assignment algorithm.

#ifndef vsip_core_parallel_assign_chain_hpp_
#define vsip_core_parallel_assign_chain_hpp_

#include <vector>
#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip/core/extdata.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/parallel/assign.hpp>
#include <vsip/core/parallel/assign_common.hpp>

#define PCA_ROTATE  0
#define PCA_VERBOSE 0

#if PCA_VERBOSE
#  include <vsip_csl/output/domain.hpp>
#endif

namespace vsip
{
namespace impl
{
namespace par_chain_assign
{

template <typename OrderT,
	  typename ExtDataT>
inline void
chain_add(
  impl::Chain_builder& builder,
  ExtDataT&            ext,
  Domain<1> const&     dom)
{
  typedef typename ExtDataT::element_type element_type;

  dimension_type const dim0 = OrderT::impl_dim0;
  assert(dim0 == 0);

  builder.add<element_type>(sizeof(element_type) * dom.first()*ext.stride(dim0),
			  dom.stride() * ext.stride(dim0),
			  dom.length());
}



template <typename OrderT,
	  typename ExtDataT>
inline void
chain_add(
  impl::Chain_builder& builder,
  ExtDataT&            ext,
  Domain<2> const&     dom)
{
  typedef typename ExtDataT::element_type element_type;

  dimension_type const dim0 = OrderT::impl_dim0;
  dimension_type const dim1 = OrderT::impl_dim1;

  builder.add<element_type>(
              sizeof(element_type) * (dom[dim0].first()*ext.stride(dim0) +
				      dom[dim1].first()*ext.stride(dim1)),
	      dom[dim0].stride() * ext.stride(dim0), dom[dim0].length(),
	      dom[dim1].stride() * ext.stride(dim1), dom[dim1].length());
}



template <typename OrderT,
	  typename ExtDataT>
inline void
chain_add(
  impl::Chain_builder& builder,
  ExtDataT&            ext,
  Domain<3> const&    dom)
{
  typedef typename ExtDataT::element_type element_type;

  dimension_type const dim0 = OrderT::impl_dim0;
  dimension_type const dim1 = OrderT::impl_dim1;
  dimension_type const dim2 = OrderT::impl_dim2;

  for (index_type i = 0; i < dom[dim0].size(); ++i)
  {
    builder.add<element_type>(
                sizeof(element_type) *
		  ( (dom[dim0].first()+i*dom[dim0].stride())*ext.stride(dim0)
		  +  dom[dim1].first()                      *ext.stride(dim1)
		  +  dom[dim2].first()                      *ext.stride(dim2)),
		dom[dim1].stride() * ext.stride(dim1), dom[dim1].length(),
		dom[dim2].stride() * ext.stride(dim2), dom[dim2].length());
  }
}




template <dimension_type Dim,
	  typename       T,
	  typename       Block,
	  typename       AppMapT,
	  typename       ExtDataT>
void
build_ext_array(
  typename View_of_dim<Dim, T, Block>::const_type view,
  AppMapT&                                      am,
  ExtDataT**                                    array,
  sync_action_type                              sync)
{
  typedef typename Distributed_local_block<Block>::type local_block_type;
  typedef typename View_of_dim<Dim, T, local_block_type>::const_type
		local_view_type;

  processor_type rank = local_processor();

  // First set all subblock ext pointers to NULL.
  length_type tot_sb = am.num_subblocks();
  for (index_type sb=0; sb<tot_sb; ++sb)
    array[sb] = NULL;

  // Then, initialize the subblocks this processor actually owns.
  index_type sb = am.subblock(rank);
  if (sb != no_subblock)
  {
    local_view_type local_view = get_local_view(view);
    array[sb] = new ExtDataT(local_view.block(), sync);
  }
}



template <typename ExtDataT>
void
cleanup_ext_array(
  length_type num_subblocks,
  ExtDataT**  array)
{
  for (index_type sb=0; sb<num_subblocks; ++sb)
    if (array[sb] != NULL)
      delete array[sb];
}

} // namespace par_chain_assign


// Chained parallel assignment.
template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
class Par_assign<Dim, T1, T2, Block1, Block2, Chained_assign>
  : Compile_time_assert<Is_split_block<Block1>::value ==
                        Is_split_block<Block2>::value>
{
  static dimension_type const dim = Dim;

  // disable_copy should only be set to true for testing purposes.  It
  // disables direct copy of data when source and destination are on
  // the same processor, causing chains to be built on both sides.
  // This is helps cover chain-to-chain copies for par-services-none.
  static bool const disable_copy = false;

  typedef typename Distributed_local_block<Block1>::type dst_local_block;
  typedef typename Distributed_local_block<Block2>::type src_local_block;

  typedef typename View_of_dim<dim, T1, dst_local_block>::type
		dst_lview_type;

  typedef typename View_of_dim<dim, T2, src_local_block>::const_type
		src_lview_type;

  typedef impl::Persistent_ext_data<src_local_block> src_ext_type;
  typedef impl::Persistent_ext_data<dst_local_block> dst_ext_type;

  typedef typename Block1::map_type dst_appmap_t;
  typedef typename Block2::map_type src_appmap_t;

  typedef typename Block_layout<Block1>::order_type dst_order_t;

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
    Msg_record(processor_type proc, index_type sb, chain_type chain)
      : proc_    (proc),
        subblock_(sb),
	chain_   (chain)
      {}

  public:
    processor_type proc_;    // destination processor
    index_type     subblock_;
    chain_type     chain_;
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
  Par_assign(
    typename View_of_dim<Dim, T1, Block1>::type       dst,
    typename View_of_dim<Dim, T2, Block2>::const_type src)
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
      src_ext_ (new src_ext_type*[src_.block().map().num_subblocks()]),
      dst_ext_ (new dst_ext_type*[dst_.block().map().num_subblocks()])
  {
    profile::Scope<profile::par> scope("Par_assign<Chained_assign>-cons");
    assert(src_am_.impl_comm() == dst_am_.impl_comm());

    par_chain_assign::build_ext_array<Dim, T2, Block2>(
      src_, src_am_, src_ext_, impl::SYNC_IN);
    par_chain_assign::build_ext_array<Dim, T1, Block1>(
      dst_, dst_am_, dst_ext_, impl::SYNC_OUT);

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

    if (send_list.size() > 0)
    {
      typedef typename std::vector<Msg_record>::iterator sl_iterator;
      sl_iterator sl_cur = send_list.begin();
      sl_iterator sl_end = send_list.end();
      for (; sl_cur != sl_end; ++sl_cur)
      {
	impl::free_chain((*sl_cur).chain_);
      }
    }

    if (recv_list.size() > 0)
    {
      typedef typename std::vector<Msg_record>::iterator rl_iterator;
      rl_iterator rl_cur = recv_list.begin();
      rl_iterator rl_end = recv_list.end();
      for (; rl_cur != rl_end; ++rl_cur)
      {
	impl::free_chain((*rl_cur).chain_);
      }
    }

    par_chain_assign::cleanup_ext_array(dst_am_.num_subblocks(), dst_ext_);
    par_chain_assign::cleanup_ext_array(src_am_.num_subblocks(), src_ext_);

    delete[] dst_ext_;
    delete[] src_ext_;
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
  typename View_of_dim<Dim, T1, Block1>::type       dst_;
  typename View_of_dim<Dim, T2, Block2>::const_type src_;

  dst_appmap_t const& dst_am_;
  src_appmap_t const& src_am_;
  impl::Communicator& comm_;

  std::vector<Msg_record>    send_list;
  std::vector<Msg_record>    recv_list;
  std::vector<Copy_record>   copy_list;

  std::vector<request_type> req_list;

  int                       msg_count;

  src_ext_type**            src_ext_;
  dst_ext_type**            dst_ext_;
};



/***********************************************************************
  Definitions
***********************************************************************/

// Build the send_list, a list of processor-subblock-local_domain
// records.  This can be done in advance of the actual assignment.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Chained_assign>::build_send_list()
{
  profile::Scope<profile::par> scope("Par_assign<Chained_assign>-build_send_list");
  processor_type rank = local_processor();

#if VSIPL_IMPL_PCA_ROTATE
  index_type  offset = src_am_.impl_rank_from_proc(rank);
#endif
  length_type dsize  = dst_am_.impl_working_size();
  // std::min(dst_am_.num_subblocks(), dst_am_.impl_pvec().size());

#if PCA_VERBOSE >= 1
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
    // Iterate over all processors
    for (index_type pi=0; pi<dsize; ++pi)
    {
      // Rotate message order so processors don't all send to 0, then 1, etc
      // (Currently does not work, it needs to take into account the
      // number of subblocks).
#if VSIPL_IMPL_PCA_ROTATE
      processor_type proc = dst_am_.impl_proc_from_rank((pi + offset) % dsize);
#else
      processor_type proc = dst_am_.impl_proc_from_rank(pi);
#endif

      // Transfers that stay on this processor is handled by the copy_list.
      if (!disable_copy && proc == rank)
	continue;

      index_type dst_sb = dst_am_.subblock(proc);

      if (dst_sb != no_subblock)
      {
	// Check to see if destination processor already has block
	if (!disable_copy && processor_has_block(src_am_, proc, src_sb))
	  continue;

	impl::Chain_builder builder;
	src_ext_type* ext = src_ext_[src_sb];
	ext->begin();

	for (index_type dp=0; dp<num_patches(dst_, dst_sb); ++dp)
	{
	  Domain<dim> dst_dom = global_domain(dst_, dst_sb, dp);

	  for (index_type sp=0; sp<num_patches(src_, src_sb); ++sp)
	  {
	    Domain<dim> src_dom  = global_domain(src_, src_sb, sp);
	    Domain<dim> src_ldom = local_domain(src_, src_sb, sp);

	    Domain<dim> intr;

	    if (intersect(src_dom, dst_dom, intr))
	    {
	      Domain<dim> send_dom = apply_intr(src_ldom, src_dom, intr);

	      par_chain_assign::chain_add<dst_order_t>(builder, *ext, send_dom);

#if PCA_VERBOSE >= 2
	      std::cout << "(" << rank << ") send "
			<< rank << "/" << src_sb << "/" << sp
			<< " -> "
			<< proc << "/" << dst_sb << "/" << dp
			<< " src: " << src_dom
			<< " dst: " << dst_dom
			<< " intr: " << intr
			<< " send: " << send_dom
		// << " val: " << get(local_view, first(send_dom))
			<< std::endl;
#endif
	    }
	  }
	}
	if (!builder.is_empty())
	  send_list.push_back(Msg_record(proc, src_sb, builder.get_chain()));
	ext->end();
      }
    }
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
Par_assign<Dim, T1, T2, Block1, Block2, Chained_assign>::build_recv_list()
{
  profile::Scope<profile::par> scope("Par_assign<Chained_assign>-build_recv_list");
  processor_type rank = local_processor();

#if VSIPL_IMPL_PCA_ROTATE
  index_type  offset = dst_am_.impl_rank_from_proc(rank);
#endif
  length_type ssize  = src_am_.impl_working_size();

#if PCA_VERBOSE >= 1
    std::cout << "(" << rank << ") "
	      << "build_recv_list(ssize: " << ssize
	      << ") -------------------------------------\n";
#endif

  index_type dst_sb = dst_am_.subblock(rank);

  if (dst_sb != no_subblock)
  {
    dst_ext_type* ext = dst_ext_[dst_sb];
    ext->begin();
      
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
      
      impl::Chain_builder builder;
      
      index_type src_sb = src_am_.subblock(proc);

      // If multiple processors have the subblock, the first processor
      // is responsible for sending it to us.

      if (src_sb != no_subblock &&
	  *(src_am_.processor_begin(src_sb)) == proc)
      {
	// Check to see if destination processor already has block
	if (!disable_copy && processor_has_block(src_am_, rank, src_sb))
	  continue;

	for (index_type dp=0; dp<num_patches(dst_, dst_sb); ++dp)
	{
	  Domain<dim> dst_dom  = global_domain(dst_, dst_sb, dp);
	  Domain<dim> dst_ldom = local_domain(dst_, dst_sb, dp);
	  
	  for (index_type sp=0; sp<num_patches(src_, src_sb); ++sp)
	  {
	    Domain<dim> src_dom = global_domain(src_, src_sb, sp);
	    
	    Domain<dim> intr;
	    
	    if (intersect(dst_dom, src_dom, intr))
	    {
	      Domain<dim> recv_dom = apply_intr(dst_ldom, dst_dom, intr);
	      
	      par_chain_assign::chain_add<dst_order_t>(builder, *ext, recv_dom);
	      
#if PCA_VERBOSE >= 1
	      std::cout << "(" << rank << ") recv "
			<< rank << "/" << dst_sb << "/" << dp
			<< " <- "
			<< proc << "/" << src_sb << "/" << sp
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
      }
      if (!builder.is_empty())
	recv_list.push_back(Msg_record(proc, dst_sb, builder.get_chain()));
    }
    ext->end();
  }
}



template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Chained_assign>::build_copy_list()
{
  profile::Scope<profile::par> scope("Par_assign<Chained_assign>-build_copy_list");
  processor_type rank = local_processor();

#if PCA_VERBOSE >= 1
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
      for (index_type dp=0; dp<num_patches(dst_, dst_sb); ++dp)
      {
	Domain<dim> dst_dom  = global_domain(dst_, dst_sb, dp);
	Domain<dim> dst_ldom = local_domain (dst_, dst_sb, dp);

	for (index_type sp=0; sp<num_patches(src_, src_sb); ++sp)
	{
	  Domain<dim> src_dom  = global_domain(src_, src_sb, sp);
	  Domain<dim> src_ldom = local_domain (src_, src_sb, sp);

	  Domain<dim> intr;

#if PCA_VERBOSE >= 2
//	  std::cout << " - dst " << dst_sb << "/" << dp << std::endl
//		    << "   src " << src_sb     << "/" << sp << std::endl
//	    ;
#endif

	  if (intersect(src_dom, dst_dom, intr))
	  {
	    Domain<dim> send_dom = apply_intr(src_ldom, src_dom, intr);
	    Domain<dim> recv_dom = apply_intr(dst_ldom, dst_dom, intr);

	    copy_list.push_back(Copy_record(src_sb, dst_sb,
					    send_dom, recv_dom));

#if PCA_VERBOSE >= 2
	    std::cout << "(" << rank << ") "
		      << "copy src: " << src_sb << "/" << sp
		      << " " << send_dom
		      << "  dst: " << dst_sb << "/" << dp
		      << " " << recv_dom
		      << std::endl
		      << "    "
		      << "src_dom: " << src_dom
		      << "  src_ldom: " << src_ldom
		      << std::endl
		      << "    "
		      << "dst_dom: " << dst_dom
		      << "  dst_ldom: " << dst_ldom
		      << std::endl
		      << "  intr: " << intr
		      << std::endl
	      ;
#endif
	  }
	}
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
Par_assign<Dim, T1, T2, Block1, Block2, Chained_assign>::exec_send_list()
{
  profile::Scope<profile::par> scope("Par_assign<Chained_assign>-exec_send_list");

#if PCA_VERBOSE >= 1
  processor_type rank = local_processor();
  std::cout << "(" << rank << ") "
	    << "exec_send_list(size: " << send_list.size()
	    << ") -------------------------------------\n";
#endif
  typedef typename std::vector<Msg_record>::iterator sl_iterator;

  {
    sl_iterator sl_cur = send_list.begin();
    sl_iterator sl_end = send_list.end();
    for (; sl_cur != sl_end; ++sl_cur)
    {
      impl::Chain_builder builder;
      processor_type proc = (*sl_cur).proc_;

      src_ext_type* ext = src_ext_[(*sl_cur).subblock_];
      ext->begin();
      builder.stitch(ext->data(), (*sl_cur).chain_);
      ext->end();
      
      chain_type chain = builder.get_chain();
      request_type   req;
      comm_.send(proc, chain, req);
      impl::free_chain(chain);
      req_list.push_back(req);
    }
  }
}



// Execute the recv_list.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Chained_assign>::exec_recv_list()
{
  profile::Scope<profile::par> scope("Par_assign<Chained_assign>-exec_recv_list");

#if PCA_VERBOSE >= 1
  processor_type rank = local_processor();
  std::cout << "(" << rank << ") "
	    << "exec_recv_list(size: " << recv_list.size()
	    << ") -------------------------------------\n";
#endif

  typedef typename std::vector<Msg_record>::iterator rl_iterator;
  rl_iterator rl_cur = recv_list.begin();
  rl_iterator rl_end = recv_list.end();
    
  for (; rl_cur != rl_end; ++rl_cur)
  {
    impl::Chain_builder builder;
    processor_type proc = (*rl_cur).proc_;

    dst_ext_type* ext = dst_ext_[(*rl_cur).subblock_];
    ext->begin();
    builder.stitch(ext->data(), (*rl_cur).chain_);
    ext->end();

    chain_type chain = builder.get_chain();
    comm_.recv(proc, chain);
    impl::free_chain(chain);
  }
}



// Execute the copy_list.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Chained_assign>::exec_copy_list()
{
  profile::Scope<profile::par> scope("Par_assign<Chained_assign>-exec_copy_list");

#if PCA_VERBOSE >= 1
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
#if PCA_VERBOSE >= 2
    std::cout << "(" << rank << ") "
	      << "src subblock: " << (*cl_cur).src_sb_ << " -> "
	      << "dst subblock: " << (*cl_cur).dst_sb_ << std::endl
	      << dst_lview((*cl_cur).dst_dom_);

    // std::cout << "  from: " << (*cl_cur).src_dom_ << std::endl;
    // std::cout << "  to  : " << (*cl_cur).dst_dom_ << std::endl;
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
Par_assign<Dim, T1, T2, Block1, Block2, Chained_assign>::wait_send_list()
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

#undef PCA_VERBOSE
#undef PCA_ROTATE

#endif

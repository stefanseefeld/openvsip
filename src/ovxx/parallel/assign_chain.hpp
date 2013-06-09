//
// Copyright (c) 2005 - 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_assign_chain_hpp_
#define ovxx_parallel_assign_chain_hpp_

#include <vector>
#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <ovxx/view/traits.hpp>
#include <ovxx/dda/persistent.hpp>
#include <ovxx/parallel/service.hpp>
#include <ovxx/parallel/assign.hpp>
#include <ovxx/parallel/get_local_view.hpp>

#define OVXX_PCA_ROTATE  0
#define OVXX_PCA_VERBOSE 0

#if OVXX_PCA_VERBOSE
//#  include <ovxx/output.hpp>
#endif

namespace ovxx
{
namespace parallel
{
namespace chain_assign
{

// FIXME: This used to come from DDA / Storage.
//        The rationale for this logic needs to be clarified,
//        especially as its use below looks bogus. 
template <typename D,
	  typename T = typename D::value_type,
	  storage_format_type S = D::layout_type::storage_format>
struct element_type
{ typedef T type;};
template <typename D, typename T>
struct element_type<D, complex<T>, interleaved_complex>
{ typedef T type;};
template <typename D, typename T>
struct element_type<D, complex<T>, split_complex>
{ typedef T type;};

template <typename O, typename Data>
inline void
chain_add(Chain_builder &builder, Data &data, Domain<1> const &dom)
{
  typedef typename element_type<Data>::type element_type;

  dimension_type const dim0 = O::impl_dim0;
  OVXX_PRECONDITION(dim0 == 0);

  builder.add<element_type>
    (sizeof(element_type) * dom.first()*data.stride(dim0),
     dom.stride() * data.stride(dim0),
     dom.length());
}

template <typename O, typename Data>
inline void
chain_add(Chain_builder &builder, Data &data, Domain<2> const &dom)
{
  typedef typename element_type<Data>::type element_type;

  dimension_type const dim0 = O::impl_dim0;
  dimension_type const dim1 = O::impl_dim1;

  builder.add<element_type>
    (sizeof(element_type) * (dom[dim0].first()*data.stride(dim0) +
			     dom[dim1].first()*data.stride(dim1)),
     dom[dim0].stride() * data.stride(dim0), dom[dim0].length(),
     dom[dim1].stride() * data.stride(dim1), dom[dim1].length());
}



template <typename O, typename Data>
inline void
chain_add(Chain_builder &builder, Data &data, Domain<3> const &dom)
{
  typedef typename element_type<Data>::type element_type;

  dimension_type const dim0 = O::impl_dim0;
  dimension_type const dim1 = O::impl_dim1;
  dimension_type const dim2 = O::impl_dim2;

  for (index_type i = 0; i < dom[dim0].size(); ++i)
  {
    builder.add<element_type>
      (sizeof(element_type) *
       ( (dom[dim0].first()+i*dom[dim0].stride())*data.stride(dim0)
	 +  dom[dim1].first()                      *data.stride(dim1)
	 +  dom[dim2].first()                      *data.stride(dim2)),
       dom[dim1].stride() * data.stride(dim1), dom[dim1].length(),
       dom[dim2].stride() * data.stride(dim2), dom[dim2].length());
  }
}

template <dimension_type D,
	  typename       Block,
	  typename       AppMapT,
	  typename       Data>
void
build_dda_array(typename view_of<Block>::const_type view,
		AppMapT &am,
		Data **array,
		dda::sync_policy sync)
{
  typedef typename ovxx::distributed_local_block<Block>::type local_block_type;
  typedef typename view_of<local_block_type>::const_type local_view_type;

  processor_type rank = local_processor();

  // First set all subblock data pointers to 0.
  length_type tot_sb = am.num_subblocks();
  for (index_type sb=0; sb<tot_sb; ++sb)
    array[sb] = 0;

  // Then, initialize the subblocks this processor actually owns.
  index_type sb = am.subblock(rank);
  if (sb != no_subblock)
  {
    local_view_type local_view = ovxx::get_local_view(view);
    array[sb] = new Data(local_view.block(), sync);
  }
}



template <typename Data>
void
cleanup_dda_array(length_type num_subblocks,
		  Data **array)
{
  for (index_type sb=0; sb<num_subblocks; ++sb)
    if (array[sb])
      delete array[sb];
}

} // namespace chain_assign


// Chained parallel assignment.
template <dimension_type D, typename LHS, typename RHS>
class Assignment<D, LHS, RHS, Chained_assign>
  : ct_assert<is_split_block<LHS>::value == is_split_block<RHS>::value>
{
  static dimension_type const dim = D;

  // disable_copy should only be set to true for testing purposes.  It
  // disables direct copy of data when source and destination are on
  // the same processor, causing chains to be built on both sides.
  // This is helps cover chain-to-chain copies for par-services-none.
  static bool const disable_copy = false;

  typedef typename ovxx::distributed_local_block<LHS>::type lhs_local_block;
  typedef typename ovxx::distributed_local_block<RHS>::type rhs_local_block;

  typedef typename view_of<lhs_local_block>::type lhs_lview_type;
  typedef typename view_of<rhs_local_block>::const_type rhs_lview_type;

  typedef ovxx::dda::Persistent_data<rhs_local_block> rhs_dda_type;
  typedef ovxx::dda::Persistent_data<lhs_local_block> lhs_dda_type;

  typedef typename LHS::map_type lhs_appmap_t;
  typedef typename RHS::map_type rhs_appmap_t;

  typedef typename get_block_layout<LHS>::order_type lhs_order_t;

  typedef Communicator::request_type request_type;
  typedef Communicator::chain_type   chain_type;

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
  ///   RHS_SB_ is the source local subblock,
  ///   LHS_SB_ is the destination local subblock,
  ///   RHS_DOM_ is the local domain within the source subblock to transfer,
  ///   LHS_DOM_ is the local domain within the destination subblock to
  ///      transfer.

  struct Copy_record
  {
    Copy_record(index_type rhs_sb, index_type lhs_sb,
	       Domain<D> rhs_dom,
	       Domain<D> lhs_dom)
      : rhs_sb_  (rhs_sb),
        lhs_sb_  (lhs_sb),
	rhs_dom_ (rhs_dom),
	lhs_dom_ (lhs_dom)
      {}

  public:
    index_type     rhs_sb_;    // destination processor
    index_type     lhs_sb_;
    Domain<D>    rhs_dom_;
    Domain<D>    lhs_dom_;
  };


  // Constructor.
public:
  Assignment(typename view_of<LHS>::type lhs,
	     typename view_of<RHS>::const_type rhs)
    : lhs_      (lhs),
      rhs_      (rhs.block()),
      lhs_am_   (lhs_.block().map()),
      rhs_am_   (rhs_.block().map()),
      comm_     (lhs_am_.impl_comm()),
      send_list (),
      recv_list (),
      copy_list (),
      req_list  (),
      msg_count (0),
      rhs_dda_ (new rhs_dda_type*[rhs_.block().map().num_subblocks()]),
      lhs_dda_ (new lhs_dda_type*[lhs_.block().map().num_subblocks()])
  {
    OVXX_PRECONDITION(rhs_am_.impl_comm() == lhs_am_.impl_comm());

    chain_assign::build_dda_array<D, RHS>(rhs_, rhs_am_, rhs_dda_, dda::in);
    chain_assign::build_dda_array<D, LHS>(lhs_, lhs_am_, lhs_dda_, dda::out);

    build_send_list();
    if (!disable_copy)
      build_copy_list();
    build_recv_list();
  }

  ~Assignment()
  {
    // At destruction, the list of outstanding sends should be empty.
    // This would be non-empty if:
    //  - Par_assign did not to clear the lists after
    //    processing it (library design error), or
    //  - User executed send() without a corresponding wait().
    OVXX_PRECONDITION(req_list.size() == 0);

    if (send_list.size() > 0)
    {
      typedef typename std::vector<Msg_record>::iterator sl_iterator;
      sl_iterator sl_cur = send_list.begin();
      sl_iterator sl_end = send_list.end();
      for (; sl_cur != sl_end; ++sl_cur)
      {
	free_chain((*sl_cur).chain_);
      }
    }

    if (recv_list.size() > 0)
    {
      typedef typename std::vector<Msg_record>::iterator rl_iterator;
      rl_iterator rl_cur = recv_list.begin();
      rl_iterator rl_end = recv_list.end();
      for (; rl_cur != rl_end; ++rl_cur)
      {
	free_chain((*rl_cur).chain_);
      }
    }

    chain_assign::cleanup_dda_array(lhs_am_.num_subblocks(), lhs_dda_);
    chain_assign::cleanup_dda_array(rhs_am_.num_subblocks(), rhs_dda_);

    delete[] lhs_dda_;
    delete[] rhs_dda_;
  }

  void operator()()
  {
    if (send_list.size() > 0) exec_send_list();
    if (copy_list.size() > 0) exec_copy_list();
    if (recv_list.size() > 0) exec_recv_list();

    if (req_list.size() > 0)  wait_send_list();

    cleanup();
  }

private:
  typename view_of<LHS>::type lhs_;
  typename view_of<RHS>::const_type rhs_;

  void build_send_list();
  void build_recv_list();
  void build_copy_list();

  void exec_send_list();
  void exec_recv_list();
  void exec_copy_list();

  void wait_send_list();

  void cleanup() {}	// Cleanup send_list buffers.

  lhs_appmap_t const& lhs_am_;
  rhs_appmap_t const& rhs_am_;
  Communicator& comm_;

  std::vector<Msg_record>    send_list;
  std::vector<Msg_record>    recv_list;
  std::vector<Copy_record>   copy_list;

  std::vector<request_type> req_list;

  int                       msg_count;

  rhs_dda_type**            rhs_dda_;
  lhs_dda_type**            lhs_dda_;
};



/***********************************************************************
  Definitions
***********************************************************************/

// Build the send_list, a list of processor-subblock-local_domain
// records.  This can be done in advance of the actual assignment.

template <dimension_type D, typename LHS, typename RHS>
void
Assignment<D, LHS, RHS, Chained_assign>::build_send_list()
{
  processor_type rank = local_processor();

#if OVXX_PCA_ROTATE
  index_type  offset = rhs_am_.impl_rank_from_proc(rank);
#endif
  length_type dsize  = lhs_am_.impl_working_size();
  // std::min(lhs_am_.num_subblocks(), lhs_am_.impl_pvec().size());

#if OVXX_PCA_VERBOSE >= 1
    std::cout << "(" << rank << ") "
	      << "build_send_list(dsize: " << dsize
	      << ") -------------------------------------\n";
#endif

  index_type rhs_sb = rhs_am_.subblock(rank);

  // If multiple processors have the subblock, the first processor
  // is responsible for sending it.

  if (rhs_sb != no_subblock &&
      *(rhs_am_.processor_begin(rhs_sb)) == rank)
  {
    // Iterate over all processors
    for (index_type pi=0; pi<dsize; ++pi)
    {
      // Rotate message order so processors don't all send to 0, then 1, etc
      // (Currently does not work, it needs to take into account the
      // number of subblocks).
#if OVXX_PCA_ROTATE
      processor_type proc = lhs_am_.impl_proc_from_rank((pi + offset) % dsize);
#else
      processor_type proc = lhs_am_.impl_proc_from_rank(pi);
#endif

      // Transfers that stay on this processor is handled by the copy_list.
      if (!disable_copy && proc == rank)
	continue;

      index_type lhs_sb = lhs_am_.subblock(proc);

      if (lhs_sb != no_subblock)
      {
	// Check to see if destination processor already has block
	if (!disable_copy && processor_has_block(rhs_am_, proc, rhs_sb))
	  continue;

	Chain_builder builder;
	rhs_dda_type* dda = rhs_dda_[rhs_sb];
	dda->sync_in();

	for (index_type dp=0; dp<num_patches(lhs_, lhs_sb); ++dp)
	{
	  Domain<dim> lhs_dom = global_domain(lhs_, lhs_sb, dp);

	  for (index_type sp=0; sp<num_patches(rhs_, rhs_sb); ++sp)
	  {
	    Domain<dim> rhs_dom  = global_domain(rhs_, rhs_sb, sp);
	    Domain<dim> rhs_ldom = local_domain(rhs_, rhs_sb, sp);

	    Domain<dim> intr;

	    if (intersect(rhs_dom, lhs_dom, intr))
	    {
	      Domain<dim> send_dom = apply_intr(rhs_ldom, rhs_dom, intr);

	      chain_assign::chain_add<lhs_order_t>(builder, *dda, send_dom);

#if OVXX_PCA_VERBOSE >= 2
	      std::cout << "(" << rank << ") send "
			<< rank << "/" << rhs_sb << "/" << sp
			<< " -> "
			<< proc << "/" << lhs_sb << "/" << dp
			<< " rhs: " << rhs_dom
			<< " lhs: " << lhs_dom
			<< " intr: " << intr
			<< " send: " << send_dom
		// << " val: " << get(local_view, first(send_dom))
			<< std::endl;
#endif
	    }
	  }
	}
	if (!builder.is_empty())
	  send_list.push_back(Msg_record(proc, rhs_sb, builder.get_chain()));
	dda->sync_out();
      }
    }
  }
}



// Build the recv_list, a list of processor-subblock-local_domain
// records.  This can be done in advance of the actual assignment.

template <dimension_type D, typename LHS, typename RHS>
void
Assignment<D, LHS, RHS, Chained_assign>::build_recv_list()
{
  processor_type rank = local_processor();

#if OVXX_PCA_ROTATE
  index_type  offset = lhs_am_.impl_rank_from_proc(rank);
#endif
  length_type ssize  = rhs_am_.impl_working_size();

#if OVXX_PCA_VERBOSE >= 1
    std::cout << "(" << rank << ") "
	      << "build_recv_list(ssize: " << ssize
	      << ") -------------------------------------\n";
#endif

  index_type lhs_sb = lhs_am_.subblock(rank);

  if (lhs_sb != no_subblock)
  {
    lhs_dda_type* dda = lhs_dda_[lhs_sb];
    dda->sync_in();
      
    // Iterate over all sending processors
    for (index_type pi=0; pi<ssize; ++pi)
    {
      // Rotate message order so processors don't all send to 0,
      // then 1, etc (Currently does not work, it needs to take into
      // account the number of subblocks).
      // processor_type proc = (rhs_am_.impl_proc_from_rank(pi) + rank) % size;
      processor_type proc = rhs_am_.impl_proc_from_rank(pi);

      // Transfers that stay on this processor is handled by the copy_list.
      if (!disable_copy && proc == rank)
	continue;
      Chain_builder builder;
      
      index_type rhs_sb = rhs_am_.subblock(proc);

      // If multiple processors have the subblock, the first processor
      // is responsible for sending it to us.

      if (rhs_sb != no_subblock &&
	  *(rhs_am_.processor_begin(rhs_sb)) == proc)
      {
	// Check to see if destination processor already has block
	if (!disable_copy && processor_has_block(rhs_am_, rank, rhs_sb))
	  continue;

	for (index_type dp=0; dp<num_patches(lhs_, lhs_sb); ++dp)
	{
	  Domain<dim> lhs_dom  = global_domain(lhs_, lhs_sb, dp);
	  Domain<dim> lhs_ldom = local_domain(lhs_, lhs_sb, dp);
	  
	  for (index_type sp=0; sp<num_patches(rhs_, rhs_sb); ++sp)
	  {
	    Domain<dim> rhs_dom = global_domain(rhs_, rhs_sb, sp);
	    
	    Domain<dim> intr;
	    
	    if (intersect(lhs_dom, rhs_dom, intr))
	    {
	      Domain<dim> recv_dom = apply_intr(lhs_ldom, lhs_dom, intr);
	      
	      chain_assign::chain_add<lhs_order_t>(builder, *dda, recv_dom);
	      
#if OVXX_PCA_VERBOSE >= 1
	      std::cout << "(" << rank << ") recv "
			<< rank << "/" << lhs_sb << "/" << dp
			<< " <- "
			<< proc << "/" << rhs_sb << "/" << sp
			<< " lhs: " << lhs_dom
			<< " rhs: " << rhs_dom
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
	recv_list.push_back(Msg_record(proc, lhs_sb, builder.get_chain()));
    }
    dda->sync_out();
  }
}



template <dimension_type D, typename LHS, typename RHS>
void
Assignment<D, LHS, RHS, Chained_assign>::build_copy_list()
{
  processor_type rank = local_processor();

#if OVXX_PCA_VERBOSE >= 1
  std::cout << "(" << rank << ") "
	    << "build_copy_list(num_procs: " << rhs_am_.num_processors()
	    << ") -------------------------------------\n";
#endif

  index_type lhs_sb = lhs_am_.subblock(rank);
  if (lhs_sb != no_subblock)
  {
    index_type rhs_sb = rhs_am_.subblock(rank);
    if (rhs_sb != no_subblock)
    {
      for (index_type dp=0; dp<num_patches(lhs_, lhs_sb); ++dp)
      {
	Domain<dim> lhs_dom  = global_domain(lhs_, lhs_sb, dp);
	Domain<dim> lhs_ldom = local_domain (lhs_, lhs_sb, dp);

	for (index_type sp=0; sp<num_patches(rhs_, rhs_sb); ++sp)
	{
	  Domain<dim> rhs_dom  = global_domain(rhs_, rhs_sb, sp);
	  Domain<dim> rhs_ldom = local_domain (rhs_, rhs_sb, sp);

	  Domain<dim> intr;

#if OVXX_PCA_VERBOSE >= 2
//	  std::cout << " - lhs " << lhs_sb << "/" << dp << std::endl
//		    << "   rhs " << rhs_sb     << "/" << sp << std::endl
//	    ;
#endif

	  if (intersect(rhs_dom, lhs_dom, intr))
	  {
	    Domain<dim> send_dom = apply_intr(rhs_ldom, rhs_dom, intr);
	    Domain<dim> recv_dom = apply_intr(lhs_ldom, lhs_dom, intr);

	    copy_list.push_back(Copy_record(rhs_sb, lhs_sb,
					    send_dom, recv_dom));

#if OVXX_PCA_VERBOSE >= 2
	    std::cout << "(" << rank << ") "
		      << "copy rhs: " << rhs_sb << "/" << sp
		      << " " << send_dom
		      << "  lhs: " << lhs_sb << "/" << dp
		      << " " << recv_dom
		      << std::endl
		      << "    "
		      << "rhs_dom: " << rhs_dom
		      << "  rhs_ldom: " << rhs_ldom
		      << std::endl
		      << "    "
		      << "lhs_dom: " << lhs_dom
		      << "  lhs_ldom: " << lhs_ldom
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

template <dimension_type D, typename LHS, typename RHS>
void
Assignment<D, LHS, RHS, Chained_assign>::exec_send_list()
{
#if OVXX_PCA_VERBOSE >= 1
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
      Chain_builder builder;
      processor_type proc = (*sl_cur).proc_;

      rhs_dda_type* dda = rhs_dda_[(*sl_cur).subblock_];
      dda->sync_in();
      // FIXME: Does this have to be non-const ?
      builder.stitch(dda->non_const_ptr(), (*sl_cur).chain_);
      dda->sync_out();
      
      chain_type chain = builder.get_chain();
      request_type   req;
      comm_.send(proc, chain, req);
      free_chain(chain);
      req_list.push_back(req);
    }
  }
}



// Execute the recv_list.

template <dimension_type D, typename LHS, typename RHS>
void
Assignment<D, LHS, RHS, Chained_assign>::exec_recv_list()
{
#if OVXX_PCA_VERBOSE >= 1
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
    Chain_builder builder;
    processor_type proc = (*rl_cur).proc_;

    lhs_dda_type* dda = lhs_dda_[(*rl_cur).subblock_];
    dda->sync_in();
    builder.stitch(dda->ptr(), (*rl_cur).chain_);
    dda->sync_out();

    chain_type chain = builder.get_chain();
    comm_.recv(proc, chain);
    free_chain(chain);
  }
}



// Execute the copy_list.

template <dimension_type D, typename LHS, typename RHS>
void
Assignment<D, LHS, RHS, Chained_assign>::exec_copy_list()
{
#if OVXX_PCA_VERBOSE >= 1
  processor_type rank = local_processor();
  std::cout << "(" << rank << ") "
	    << "exec_copy_list(size: " << copy_list.size()
	    << ") -------------------------------------\n";
#endif

  rhs_lview_type rhs_lview = ovxx::get_local_view(rhs_);
  lhs_lview_type lhs_lview = ovxx::get_local_view(lhs_);

  typedef typename std::vector<Copy_record>::iterator cl_iterator;
  for (cl_iterator cl_cur = copy_list.begin();
       cl_cur != copy_list.end();
       ++cl_cur)
  {
    view_assert_local(rhs_, (*cl_cur).rhs_sb_);
    view_assert_local(lhs_, (*cl_cur).lhs_sb_);

    lhs_lview((*cl_cur).lhs_dom_) = rhs_lview((*cl_cur).rhs_dom_);
#if OVXX_PCA_VERBOSE >= 2
    std::cout << "(" << rank << ") "
	      << "rhs subblock: " << (*cl_cur).rhs_sb_ << " -> "
	      << "lhs subblock: " << (*cl_cur).lhs_sb_ << std::endl
	      << lhs_lview((*cl_cur).lhs_dom_);

    // std::cout << "  from: " << (*cl_cur).rhs_dom_ << std::endl;
    // std::cout << "  to  : " << (*cl_cur).lhs_dom_ << std::endl;
#endif
  }
}



// Wait for the send_list instructions to be completed.

template <dimension_type D, typename LHS, typename RHS>
void
Assignment<D, LHS, RHS, Chained_assign>::wait_send_list()
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

} // namespace ovxx::parallel
} // namespace ovxx

#undef OVXX_PCA_VERBOSE
#undef OVXX_PCA_ROTATE

#endif

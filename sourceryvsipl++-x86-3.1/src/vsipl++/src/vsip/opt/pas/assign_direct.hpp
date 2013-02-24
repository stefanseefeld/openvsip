/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/pas/assign_direct.hpp
    @author  Jules Bergmann
    @date    2005-08-21
    @brief   VSIPL++ Library: Direct PAS parallel assignment algorithm.

*/

#ifndef VSIP_OPT_PAS_ASSIGN_DIRECT_HPP
#define VSIP_OPT_PAS_ASSIGN_DIRECT_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

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
#include <vsip/core/parallel/assign_common.hpp>
#include <vsip/opt/pas/offset.hpp>

#define VSIP_IMPL_PCA_VERBOSE 0



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace par_chain_assign
{

template <typename MsgRecord>
void
msg_push(
  std::vector<MsgRecord>& list,
  processor_type          proc,
  index_type              src_msg_offset,
  index_type              dst_msg_offset,
  stride_type             src_pbuf_offset,
  stride_type             dst_pbuf_offset,
  length_type             size)
{
  list.push_back(MsgRecord(proc,
			   src_msg_offset + src_pbuf_offset,
			   dst_msg_offset + dst_pbuf_offset,
			   size));
}

template <typename MsgRecord>
void
msg_push(
  std::vector<MsgRecord>&             list,
  processor_type                      proc,
  index_type                          src_msg_offset,
  index_type                          dst_msg_offset,
  std::pair<stride_type, stride_type> src_pbuf_offset,
  std::pair<stride_type, stride_type> dst_pbuf_offset,
  length_type                         size)
{
  list.push_back(MsgRecord(proc,
			   src_msg_offset + src_pbuf_offset.first,
			   dst_msg_offset + dst_pbuf_offset.first,
			   size));
  list.push_back(MsgRecord(proc,
			   src_msg_offset + src_pbuf_offset.second,
			   dst_msg_offset + dst_pbuf_offset.second,
			   size));
}



template <typename OrderT,
	  typename MsgRecord,
	  typename SrcExtDataT,
	  typename DstExtDataT,
	  typename OffsetT>
inline void
msg_add(
  std::vector<MsgRecord>& list,
  SrcExtDataT&            src_ext,
  DstExtDataT&            dst_ext,
  processor_type          proc,
  Domain<1> const&        src_dom,
  Domain<1> const&        dst_dom,
  Domain<1> const&        intr,
  stride_type             src_offset,
  stride_type             dst_offset,
  OffsetT                 src_comp_offset,
  OffsetT                 dst_comp_offset)
{
  dimension_type const dim  = 1;
  dimension_type const dim0 = OrderT::impl_dim0;
  assert(dim0 == 0);

  stride_type src_stride = src_ext.stride(dim0);
  stride_type dst_stride = dst_ext.stride(dim0);

  Index<dim>  src_index = first(intr) - first(src_dom);
  Index<dim>  dst_index = first(intr) - first(dst_dom);

  index_type src_msg_offset = src_index[0] * src_stride + src_offset;
  index_type dst_msg_offset = dst_index[0] * dst_stride + dst_offset;

  length_type size = intr.length();

  if (intr.stride() == 1 && src_stride == 1 && dst_stride == 1)
  {
    msg_push(list, proc,
	     src_msg_offset, dst_msg_offset,
	     src_comp_offset, dst_comp_offset,
	     size);
  }
  else
  {
    for (index_type i=0; i<size; ++i)
    {
      msg_push(list, proc,
	       src_msg_offset + i*src_stride*intr.stride(),
	       dst_msg_offset + i*dst_stride*intr.stride(),
	       src_comp_offset, dst_comp_offset,
	       1);
    }
  }
}


template <typename OrderT,
	  typename MsgRecord,
	  typename SrcExtDataT,
	  typename DstExtDataT,
	  typename OffsetT>
inline void
msg_add(
  std::vector<MsgRecord>& list,
  SrcExtDataT&            src_ext,
  DstExtDataT&            dst_ext,
  processor_type          proc,
  Domain<2> const&        src_dom,
  Domain<2> const&        dst_dom,
  Domain<2> const&        intr,
  stride_type             src_offset,
  stride_type             dst_offset,
  OffsetT                 src_comp_offset,
  OffsetT                 dst_comp_offset)
{
  dimension_type const dim = 2;

  dimension_type const dim0 = OrderT::impl_dim0;
  dimension_type const dim1 = OrderT::impl_dim1;

  Index<dim>  src_index = first(intr) - first(src_dom);
  Index<dim>  dst_index = first(intr) - first(dst_dom);

  index_type src_msg_offset = src_index[dim0] * src_ext.stride(dim0) 
                            + src_index[dim1] * src_ext.stride(dim1)
                            + src_offset;
  index_type dst_msg_offset = dst_index[dim0] * dst_ext.stride(dim0) 
                            + dst_index[dim1] * dst_ext.stride(dim1)
                            + dst_offset;

  length_type size = intr[dim1].length();

  for (index_type i=0; i<intr[dim0].size(); ++i)
  {
    if (intr[dim1].stride() == 1 && src_ext.stride(dim1) == 1 &&
	dst_ext.stride(dim1) == 1)
    {
      msg_push(list, proc,
	       src_msg_offset, dst_msg_offset,
	       src_comp_offset, dst_comp_offset,
	       size);
    }
    else
    {
      for (index_type j=0; j<size; ++j)
      {
	msg_push(list, proc,
		 src_msg_offset + j*src_ext.stride(dim1)*intr[dim1].stride(),
		 dst_msg_offset + j*dst_ext.stride(dim1)*intr[dim1].stride(),
		 src_comp_offset, dst_comp_offset,
		 1);
      }
    }
    
    src_msg_offset += intr[dim0].stride() * src_ext.stride(dim0);
    dst_msg_offset += intr[dim0].stride() * dst_ext.stride(dim0);
  }
}



template <typename OrderT,
	  typename MsgRecord,
	  typename SrcExtDataT,
	  typename DstExtDataT>
inline void
msg_add(
  std::vector<MsgRecord>& list,
  SrcExtDataT&            src_ext,
  DstExtDataT&            dst_ext,
  processor_type          proc,
  Domain<3> const&        src_dom,
  Domain<3> const&        dst_dom,
  Domain<3> const&        intr,
  stride_type             src_offset,
  stride_type             dst_offset,
  long                    src_comp_offset,
  long                    dst_comp_offset)
{
  dimension_type const dim = 3;

  dimension_type const dim0 = OrderT::impl_dim0;
  dimension_type const dim1 = OrderT::impl_dim1;
  dimension_type const dim2 = OrderT::impl_dim2;

  Index<dim>  src_index = first(intr) - first(src_dom);
  Index<dim>  dst_index = first(intr) - first(dst_dom);

  index_type src_msg_offset = src_index[dim0] * src_ext.stride(dim0) 
                            + src_index[dim1] * src_ext.stride(dim1)
                            + src_index[dim2] * src_ext.stride(dim2)
                            + src_offset;
  index_type dst_msg_offset = dst_index[dim0] * dst_ext.stride(dim0) 
                            + dst_index[dim1] * dst_ext.stride(dim1)
                            + dst_index[dim2] * dst_ext.stride(dim2)
                            + dst_offset;

  length_type size = intr[dim2].length();

  for (index_type i=0; i<intr[dim0].size(); ++i)
  {
    index_type src_msg_offset_1 = src_msg_offset;
    index_type dst_msg_offset_1 = dst_msg_offset;

    for (index_type j=0; j<intr[dim1].size(); ++j)
    {
      if (intr[dim2].stride() == 1 && src_ext.stride(dim2) == 1 &&
	  dst_ext.stride(dim2) == 1)
      {
	msg_push(list, proc,
		 src_msg_offset_1, dst_msg_offset_1,
		 src_comp_offset, dst_comp_offset,
		 size);
      }
      else
      {
	for (index_type k=0; k<size; ++k)
	{
	  msg_push(list, proc,
		src_msg_offset_1 + k*src_ext.stride(dim2)*intr[dim2].stride(),
		dst_msg_offset_1 + k*dst_ext.stride(dim2)*intr[dim2].stride(),
		src_comp_offset, dst_comp_offset,
		1);
	}
      }
    
      src_msg_offset_1 += intr[dim1].stride() * src_ext.stride(dim1);
      dst_msg_offset_1 += intr[dim1].stride() * dst_ext.stride(dim1);
    }
    src_msg_offset += intr[dim0].stride() * src_ext.stride(dim0);
    dst_msg_offset += intr[dim0].stride() * dst_ext.stride(dim0);
  }
}

} // namespace par_chain_assign


// Chained parallel assignment.
template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
class Par_assign<Dim, T1, T2, Block1, Block2, Direct_pas_assign>
  : Compile_time_assert<is_split_block<Block1>::value ==
                        is_split_block<Block2>::value>
{
  static dimension_type const dim = Dim;

  static long const ready_sem_index_ = 0;
  static long const done_sem_index_  = 0;

  // disable_copy should only be set to true for testing purposes.  It
  // disables direct copy of data when source and destination are on
  // the same processor, causing chains to be built on both sides.
  // This is helps cover chain-to-chain copies for par-services-none.
  static bool const disable_copy = false;

  typedef typename Distributed_local_block<Block1>::type dst_local_block;
  typedef typename Distributed_local_block<Block2>::type src_local_block;

  typedef typename view_of<dst_local_block>::type dst_lview_type;
  typedef typename view_of<src_local_block>::const_type src_lview_type;

  typedef impl::Persistent_ext_data<src_local_block> src_ext_type;
  typedef impl::Persistent_ext_data<dst_local_block> dst_ext_type;

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
    Msg_record(processor_type proc, index_type src_offset,
	       index_type dst_offset, length_type size)
      : proc_       (proc),
        src_offset_ (src_offset),
        dst_offset_ (dst_offset),
        size_       (size)
      {}

  public:
    processor_type proc_;    // destination processor
    index_type     src_offset_;
    index_type     dst_offset_;
    length_type    size_;

    // index_type     subblock_;
    // chain_type     chain_;
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
  Par_assign(typename view_of<Block1>::type dst,
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
    profile::Scope<profile::par> scope ("Par_assign<Direct_pas_assign>-cons");
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
    PAS_id src_pset = src_.block().map().impl_ll_pset();
    PAS_id dst_pset = dst_.block().map().impl_ll_pset();

    if (pas_pset_is_member(dst_pset))
      pas::semaphore_give(src_pset, 0);

    if (pas_pset_is_member(src_pset))
    {
      pas::semaphore_take(dst_pset, 0);
      exec_send_list();
    }

    if (copy_list.size() > 0) exec_copy_list();
    exec_recv_list();

    if (req_list.size() > 0)  wait_send_list();

    cleanup();

#if VSIP_IMPL_PCA_VERBOSE >= 1
    std::cout << "[" << local_processor() << "] assignment -- DONE\n"
	      << std::flush;
#endif
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

// Build the send_list, a list of processor-subblock-local_domain
// records.  This can be done in advance of the actual assignment.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Direct_pas_assign>::build_send_list()
{
  profile::Scope<profile::par>
    scope("Par_assign<Direct_pas_assign>-build_send_list");
  processor_type rank = local_processor();

  length_type dsize  = dst_am_.impl_working_size();
  // std::min(dst_am_.num_subblocks(), dst_am_.impl_pvec().size());

  typedef typename Offset<typename get_block_layout<Block1>::storage_format,
                          T1>::type dst_offset_type;
  typedef typename Offset<typename get_block_layout<Block2>::storage_format,
                          T2>::type src_offset_type;

  stride_type src_offset = src_.local().block().impl_offset();
  src_offset_type src_comp_offset = block_root(src_.block())
                                        .impl_component_offset();
  dst_offset_type dst_comp_offset = block_root(dst_.block())
                                        .impl_component_offset();

#if VSIP_IMPL_PCA_VERBOSE >= 1
    std::cout << "[" << rank << "] "
	      << "build_send_list(dsize: " << dsize
	      << ") -------------------------------------\n"
	      << std::flush;
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
      processor_type proc = dst_am_.impl_proc_from_rank(pi);

      // Transfers that stay on this processor are handled by the copy_list.
      if (!disable_copy && proc == rank)
	continue;

      index_type dst_sb = dst_am_.subblock(proc);

      if (dst_sb != no_subblock)
      {
	// Check to see if destination processor already has block
	if (!disable_copy && processor_has_block(src_am_, proc, src_sb))
	  continue;

	src_ext_.begin();

	
	typedef typename Distributed_local_block<Block1>::proxy_type
	  proxy_local_block_type;

	proxy_local_block_type proxy = get_local_proxy(dst_.block(), dst_sb);
	dda::Data<proxy_local_block_type> proxy_ext(proxy);
	stride_type dst_offset = proxy.impl_offset();

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
	      par_chain_assign::msg_add<dst_order_t, Msg_record>(
		send_list,
		src_ext_,
		proxy_ext,
		proc,
		src_dom, dst_dom, intr,
		src_offset, dst_offset,
		src_comp_offset,
		dst_comp_offset);

#if VSIP_IMPL_PCA_VERBOSE >= 2
	      std::cout << "(" << rank << ") send "
			<< rank << "/" << src_sb << "/" << sp
			<< " -> "
			<< proc << "/" << dst_sb << "/" << dp
			<< " src: " << src_dom
			<< " dst: " << dst_dom
			<< " intr: " << intr
			<< std::endl
			<< std::flush;
#endif
	    }
	  }
	}
	src_ext_.end();
      }
    }
  }
#if VSIP_IMPL_PCA_VERBOSE >= 1
    std::cout << "(" << rank << ") "
	      << "build_send_list DONE "
	      << " -------------------------------------\n"
	      << std::flush;
#endif
}



// Build the recv_list, a list of processor-subblock-local_domain
// records.  This can be done in advance of the actual assignment.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Direct_pas_assign>::build_recv_list()
{
  profile::Scope<profile::par> 
    scope("Par_assign<Direct_pas_assign>-build_recv_list");
}



template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Direct_pas_assign>::build_copy_list()
{
  profile::Scope<profile::par>
    scope("Par_assign<Direct_pas_assign>-build_copy_list");
  processor_type rank = local_processor();

#if VSIP_IMPL_PCA_VERBOSE >= 1
  std::cout << "(" << rank << ") "
	    << "build_copy_list(num_procs: " << src_am_.num_processors()
	    << ") -------------------------------------\n"
	    << std::flush;
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

#if VSIP_IMPL_PCA_VERBOSE >= 2
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

#if VSIP_IMPL_PCA_VERBOSE >= 2
	    std::cout << "(" << rank << ")"
		      << "copy src: " << src_sb << "/" << sp
		      << " " << send_dom
		      << "  dst: " << dst_sb << "/" << dp
		      << " " << recv_dom
		      << std::endl
		      << std::flush;
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
Par_assign<Dim, T1, T2, Block1, Block2, Direct_pas_assign>::exec_send_list()
{
  profile::Scope<profile::par> 
    scope("Par_assign<Direct_pas_assign>-exec_send_list");

  PAS_id src_pnum          = local_processor();
  PAS_pbuf_handle src_pbuf = src_.block().impl_ll_pbuf();
  PAS_pbuf_handle dst_pbuf = dst_.block().impl_ll_pbuf();
  PAS_id dst_pset = dst_.block().map().impl_ll_pset();
  long pull_flags = PAS_WAIT;
  long rc;

#if VSIP_IMPL_PCA_VERBOSE >= 1
  std::cout << "(" << local_processor() << ") "
	    << "exec_send_list(size: " << send_list.size()
	    << ") -------------------------------------\n"
	    << std::flush;
#endif
  typedef typename std::vector<Msg_record>::iterator sl_iterator;

  size_t elem_size;

  if (is_split_block<Block1>::value)
    elem_size = sizeof(typename scalar_of<T1>::type);
  else
    elem_size = sizeof(T1);

  sl_iterator msg    = send_list.begin();
  sl_iterator sl_end = send_list.end();
  if (msg != sl_end)
  {
    for (; msg != sl_end; ++msg)
    {
      rc = pas_move_nbytes(
	src_pnum,
	src_pbuf,
	elem_size*(*msg).src_offset_,
	(*msg).proc_,		// dst_pnum
	dst_pbuf,
	elem_size*(*msg).dst_offset_,
	elem_size*(*msg).size_,
	0,
	pull_flags | PAS_PUSH | VSIP_IMPL_PAS_XFER_ENGINE,
	NULL);
      VSIP_IMPL_CHECK_RC(rc, "pas_move_nbytes");
    }
  }
  pas::semaphore_give(dst_pset, ready_sem_index_);
}



// Execute the recv_list.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Direct_pas_assign>::exec_recv_list()
{
  profile::Scope<profile::par>
    scope("Par_assign<Direct_pas_assign>-exec_recv_list");
  PAS_id dst_pset = dst_.block().map().impl_ll_pset();
  PAS_id src_pset = src_.block().map().impl_ll_pset();

#if VSIP_IMPL_PCA_VERBOSE >= 1
  processor_type rank = local_processor();
  std::cout << "(" << rank << ") "
	    << "exec_recv_list(size: " << recv_list.size()
	    << ") -------------------------------------\n"
	    << std::flush;
#endif

  if (pas_pset_is_member(dst_pset))
    pas::semaphore_take(src_pset, ready_sem_index_);
}



// Execute the copy_list.

template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2>
void
Par_assign<Dim, T1, T2, Block1, Block2, Direct_pas_assign>::exec_copy_list()
{
  profile::Scope<profile::par>
    scope("Par_assign<Direct_pas_assign>-exec_copy_list");

#if VSIP_IMPL_PCA_VERBOSE >= 1
  processor_type rank = local_processor();
  std::cout << "(" << rank << ") "
	    << "exec_copy_list(size: " << copy_list.size()
	    << ") -------------------------------------\n"
	    << std::flush;
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
#if VSIP_IMPL_PCA_VERBOSE >= 2
    std::cout << "(" << rank << ") "
	      << "src subblock: " << (*cl_cur).src_sb_ << " -> "
	      << "dst subblock: " << (*cl_cur).dst_sb_ << std::endl
	      << dst_lview((*cl_cur).dst_dom_)
	      << std::flush;
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
Par_assign<Dim, T1, T2, Block1, Block2, Direct_pas_assign>::wait_send_list()
{
}



} // namespace vsip::impl

} // namespace vsip

#undef VSIP_IMPL_PCA_VERBOSE

#endif

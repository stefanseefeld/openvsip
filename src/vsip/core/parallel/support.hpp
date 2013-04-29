//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_PARALLEL_SUPPORT_HPP
#define VSIP_CORE_PARALLEL_SUPPORT_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/parallel/services.hpp>
#include <vsip/support.hpp>
#include <vsip/core/vector.hpp>
#include <vsip/core/domain_utils.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

inline const_Vector<processor_type> processor_set()
{
  static Dense<1, processor_type> *pset_block_ = 0;

  if (pset_block_ == 0)
  {
    impl::Communicator::pvec_type const& 
      pvec = impl::default_communicator().pvec(); 

    pset_block_ = new Dense<1, processor_type>(Domain<1>(pvec.size()));
    for (index_type i=0; i<pvec.size(); ++i)
      pset_block_->put(i, pvec[i]);
  }

  return Vector<processor_type>(*pset_block_);
}

namespace impl
{

namespace psf_detail
{

// Return the subdomain of a view/map pair for a subblock.

template <typename ViewT>
inline Domain<ViewT::dim>
subblock_domain(
  ViewT const&        view,
  Local_map const&    /*map*/,
  index_type          sb)
{
  assert(sb == 0 || sb == no_subblock);
  return (sb == 0) ? block_domain<ViewT::dim>(view.block())
                   : empty_domain<ViewT::dim>();
}

template <typename ViewT,
	  typename MapT>
inline Domain<ViewT::dim>
subblock_domain(
  ViewT const&     /*view*/,
  MapT const&      map,
  index_type       sb)
{
  return map.template impl_subblock_domain<ViewT::dim>(sb);
}



// Return the local domain of a view/map pair for a subblock/patch.

template <typename ViewT>
inline Domain<ViewT::dim>
local_domain(
  ViewT const& view,
  Local_map const&    /*map*/,
  index_type          sb,
  index_type          p)
{
  assert((sb == 0 && p == 0) || sb == no_subblock);
  return (sb == 0) ? block_domain<ViewT::dim>(view.block())
                   : empty_domain<ViewT::dim>();
}

template <typename ViewT,
	  typename MapT>
inline Domain<ViewT::dim>
local_domain(
  ViewT const&     /*view*/,
  MapT const&      map,
  index_type       sb,
  index_type       p)
{
  return map.template impl_local_domain<ViewT::dim>(sb, p);
}



// Return the global domain of a view/map pair for a subblock/patch.

template <typename ViewT>
inline Domain<ViewT::dim>
global_domain(
  ViewT const& view,
  Local_map const&    /*map*/,
  index_type          sb,
  index_type          p ATTRIBUTE_UNUSED)
{
  assert((sb == 0 && p == 0) || sb == no_subblock);
  return (sb == 0) ? block_domain<ViewT::dim>(view.block())
                   : empty_domain<ViewT::dim>();
}

template <typename ViewT,
	  typename MapT>
inline Domain<ViewT::dim>
global_domain(
  ViewT const&     /*view*/,
  MapT const&      map,
  index_type       sb,
  index_type       p)
{
  return map.template impl_global_domain<ViewT::dim>(sb, p);
}

} // namespace vsip::impl::psf_detail
} // namespace vsip::impl



/***********************************************************************
  Definitions - [view.support.fcn] parallel support functions
***********************************************************************/

/// Return the domain of VIEW's subblock SB.

/// Requires
///   VIEW to be a view
///   SB to either be a valid subblock of VIEW, or the value no_subblock.
///
/// Returns
///   The domain of VIEW's subblock SB if SB is valid, the empty
///   domain if SB == no_subblock.

template <typename ViewT>
Domain<ViewT::dim>
subblock_domain(
  ViewT const&  view,
  index_type    sb)
{
  return impl::psf_detail::subblock_domain(view, view.block().map(), sb);
}



/// Return the domain of VIEW's subblock held by the local processor.

/// Requires
///   VIEW to be a view
///
/// Returns
///   The domain of VIEW's subblock held by the local processor.

template <typename ViewT>
Domain<ViewT::dim>
subblock_domain(
  ViewT const&  view)
{
  return impl::psf_detail::subblock_domain(view, view.block().map(),
					   view.block().map().subblock());
}



/// Return the local domain of VIEW's subblock SB patch P

/// Requires
///   VIEW to be a view
///   SB to either be a valid subblock of VIEW, or the value no_subblock.
///   P to either be a valid patch of subblock SB.
///
/// Returns
///   The local domain of VIEW's subblock SB patch P if SB is valid,
///   the empty domain if SB == no_subblock.

template <typename ViewT>
Domain<ViewT::dim>
local_domain(
  ViewT const&  view,
  index_type    sb,
  index_type    p)
{
  return impl::psf_detail::local_domain(view, view.block().map(), sb, p);
}



/// Return the local domain of VIEW's patch P on the local processor's subblock

/// Requires
///   VIEW to be a view
///   P to either be a valid patch of the local processor's subblock.
///
/// Returns
///   The local domain of VIEW's patch P of the local processor's subblock
///     if the local processor holds a subblock,
///   The empty domain otherwise.

template <typename ViewT>
Domain<ViewT::dim>
local_domain(
  ViewT const&  view,
  index_type    p=0)
{
  return impl::psf_detail::local_domain(view, view.block().map(),
					view.block().map().subblock(),
					p);
}



/// Return the global domain of VIEW's subblock SB patch P

/// Requires
///   VIEW to be a view
///   SB to either be a valid subblock of VIEW, or the value no_subblock.
///   P to either be a valid patch of subblock SB.
///
/// Returns
///   The global domain of VIEW's subblock SB patch P if SB is valid,
///   the empty domain if SB == no_subblock.

template <typename ViewT>
Domain<ViewT::dim>
global_domain(
  ViewT const&  view,
  index_type    sb,
  index_type    p)
{
  return impl::psf_detail::global_domain(view, view.block().map(), sb, p);
}



/// Return the global domain of VIEW's local subblock patch P

/// Requires
///   VIEW to be a view
///   P to either be a valid patch of the local processor's subblock.
///
/// Returns
///   The global domain of VIEW's patch P of the local processor's subblock
///     if the local processor holds a subblock,
///   The empty domain otherwise.

template <typename ViewT>
Domain<ViewT::dim>
global_domain(
  ViewT const&  view,
  index_type    p=0)
{
  return impl::psf_detail::global_domain(view, view.block().map(), 
					 view.block().map().subblock(),
					 p);
}


/// Return the number of subblocks VIEW is distrubted over.

/// Requires
///   VIEW to be a view

template <typename ViewT>
length_type
num_subblocks(
  ViewT const&  view)
{
  return view.block().map().num_subblocks();
}



/// Return the number of patches in VIEW's subblock SB.

/// Requires
///   VIEW to be a view.
///   SB to either be a valid subblock of VIEW, or the value no_subblock.

template <typename ViewT>
length_type
num_patches(
  ViewT const&  view,
  index_type    sb)
{
  return view.block().map().impl_num_patches(sb);
}



/// Return the number of patches in VIEW's subblock held on the local
/// processor.

/// Requires
///   VIEW to be a view.

template <typename ViewT>
length_type
num_patches(
  ViewT const&  view)
{
  return view.block().map().impl_num_patches(view.block().map().subblock());
}



/// Return the subblock rank held by processor PR.

/// Requires
///   VIEW to be a view.
///   PR to be processor.
///
/// Returns
///   The subblock rank of VIEW held by processor PR if it holds a subblock,
///   NO_SUBBLOCK otherwise.

template <typename ViewT>
index_type
subblock(
  ViewT const&   view,
  processor_type pr)
{
  return view.block().map().subblock(pr);
}



/// Return the subblock rank held by processor PR.

/// Requires
///   VIEW to be a view.
///
/// Returns
///   The subblock rank of VIEW held by local processor,
///   or NO_SUBBLOCK if it does not hold a subblock.

template <typename ViewT>
index_type
subblock(
  ViewT const&   view)
{
  return view.block().map().subblock();
}



/// Determine which subblock holds VIEW's global index IDX

template <typename ViewT>
index_type
subblock_from_global_index(
  ViewT const&             view,
  Index<ViewT::dim> const& idx)
{
  for (dimension_type d=0; d<ViewT::dim; ++d)
    assert(idx[d] < view.size(d));

  return view.block().
    map().template impl_subblock_from_global_index<ViewT::dim>(idx);
}



/// Determine which patch holds VIEW's global index IDX

/// Notes:
///   This patch is only valid in the subblock returned by
///   subblock_from_global_index.

template <typename ViewT>
index_type
patch_from_global_index(
  ViewT const&             view,
  Index<ViewT::dim> const& idx)
{
  for (dimension_type d=0; d<ViewT::dim; ++d)
    assert(idx[d] < view.size(d));

  return view.block().
    map().template impl_subblock_from_global_index<ViewT::dim>(idx);
}



/***********************************************************************
  local_from_global_index
***********************************************************************/

/// Determine the local index corresponding to VIEW's global index G_IDX.

/// Notes:
///   This local index is only valid in processors hold the subblock
///   returned by subblock_from_global_index.

template <typename ViewT>
Index<ViewT::dim>
local_from_global_index(
  ViewT const&             view,
  Index<ViewT::dim> const& g_idx)
VSIP_NOTHROW
{
  Index<ViewT::dim> l_idx;

  for (dimension_type d=0; d<ViewT::dim; ++d)
    l_idx[d] = local_from_global_index(view, d, g_idx[d]);

  return l_idx;
}



/// Determine the local index corresponding to VIEW's global index
/// G_IDX for dimension DIM.

template <typename ViewT>
index_type
local_from_global_index(
  ViewT const&             view,
  dimension_type           dim,
  index_type               g_idx)
VSIP_NOTHROW
{
  return view.block().map().impl_local_from_global_index(dim, g_idx);
}



/***********************************************************************
  global_from_local_index
***********************************************************************/
/// Determine BLOCKS's global index corresponding to local index L_IDX

template <typename Block, dimension_type Dim>
inline
Index<Dim>
global_from_local_index_blk(
  Block const&             b,
  Index<Dim> const& l_idx)
{
  Index<Dim> g_idx;

  for (dimension_type d=0; d<Dim; ++d)
    g_idx[d] = 
      b.map().impl_global_from_local_index(d, b.map().subblock(), l_idx[d]);

  return g_idx;
}


/// Determine VIEW's global index corresponding to local index L_IDX
/// of subblock SB.

template <typename ViewT>
inline
Index<ViewT::dim>
global_from_local_index(
  ViewT const&             view,
  index_type               sb,
  Index<ViewT::dim> const& l_idx)
{
  Index<ViewT::dim> g_idx;

  for (dimension_type d=0; d<ViewT::dim; ++d)
    g_idx[d] = global_from_local_index(view, d, sb, l_idx[d]);

  return g_idx;
}



/// Determine VIEW's global index corresponding to local index L_IDX
/// of the subblock held on the local processor.

template <typename ViewT>
inline
Index<ViewT::dim>
global_from_local_index(
  ViewT const&             view,
  Index<ViewT::dim> const& l_idx)
{
  Index<ViewT::dim> g_idx;

  for (dimension_type d=0; d<ViewT::dim; ++d)
    g_idx[d] = global_from_local_index(view, d, l_idx[d]);

  return g_idx;
}



/// Determine VIEW's global index corresponding to local index L_IDX
/// for dimension DIM of subblock SB.

template <typename ViewT>
inline
index_type
global_from_local_index(
  ViewT const&   view,
  dimension_type dim,
  index_type     sb,
  index_type     l_idx)
{
  return view.block().map().impl_global_from_local_index(dim, sb, l_idx);
}



/// Determine VIEW's global index corresponding to local index L_IDX
/// for dimension DIM of the subblock held on the local processor.

template <typename ViewT>
inline
index_type
global_from_local_index(
  ViewT const&   view,
  dimension_type dim,
  index_type     l_idx)
{
  index_type sb = view.block().map().subblock();

  if (sb != no_subblock)
    return view.block().map().impl_global_from_local_index(dim, sb, l_idx);
  else
    return no_index;
}

} // namespace vsip

#endif // VSIP_IMPL_PAR_SUPPORT_HPP

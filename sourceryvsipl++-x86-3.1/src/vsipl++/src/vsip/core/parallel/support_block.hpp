/* Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved. */

/** @file    vsip/core/parallel/support_block.hpp
    @author  Jules Bergmann
    @date    2007-05-07
    @brief   VSIPL++ Library: Block versions of parallel support funcions.

*/

#ifndef VSIP_CORE_PARALLEL_SUPPORT_BLOCK_HPP
#define VSIP_CORE_PARALLEL_SUPPORT_BLOCK_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/parallel/services.hpp>
#include <vsip/support.hpp>
#include <vsip/core/domain_utils.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace psf_detail
{

// Return the subdomain of a block/map pair for a subblock.

// Variant of subblock_domain that works with blocks instead of views.
// This could be used from vsip::subblock_domain.  However, first need
// to verify that pushing the view.block() into vsip::subblock_domain
// doesn't force it to always be called.

template <dimension_type Dim,
	  typename       BlockT>
inline Domain<Dim>
block_subblock_domain(
  BlockT const&       block,
  Local_map const&    /*map*/,
  index_type          sb)
{
  assert(sb == 0 || sb == no_subblock);
  return (sb == 0) ? block_domain<Dim>(block)
                   : empty_domain<Dim>();
}

template <dimension_type Dim,
	  typename       BlockT,
	  typename       MapT>
inline Domain<Dim>
block_subblock_domain(
  BlockT const&    /*block*/,
  MapT const&      map,
  index_type       sb)
{
  return map.template impl_subblock_domain<Dim>(sb);
}

} // namespace vsip::impl::psf_detail



/***********************************************************************
  Definitions - Sourcery VSIPL++ extended parallel support functions
***********************************************************************/

/// Return the domain of BLOCK's subblock SB.

/// Requires
///   DIM to be dimension of block,
///   BLOCK to be a block,
///   SB to either be a valid subblock of BLOCK, or the value no_subblock.
///
/// Returns
///   The domain of BLOCK's subblock SB if SB is valid, the empty
///   domain if SB == no_subblock.

template <dimension_type Dim,
	  typename       BlockT>
Domain<Dim>
block_subblock_domain(
  BlockT const& block,
  index_type    sb)
{
  return impl::psf_detail::block_subblock_domain<Dim, BlockT>(
    block, block.map(), sb);
}



/// Return the domain of BLOCK's subblock held by the local processor.

/// Requires
///   BLOCK to be a view
///
/// Returns
///   The domain of BLOCK's subblock held by the local processor.

template <dimension_type Dim,
	  typename       BlockT>
Domain<Dim>
block_subblock_domain(
  BlockT const& block)
{
  return impl::psf_detail::block_subblock_domain<Dim, BlockT>(
    block, block.map(), block.map().subblock());
}

} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_IMPL_PAR_SUPPORT_BLOCK_HPP

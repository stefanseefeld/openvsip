//
// Copyright (c) 2007 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_support_hpp_
#define ovxx_parallel_support_hpp_

#include <ovxx/parallel/service.hpp>
#include <vsip/support.hpp>
#include <ovxx/domain_utils.hpp>

namespace ovxx
{
namespace parallel
{
// Return the subdomain of a block/map pair for a subblock.

// Variant of subblock_domain that works with blocks instead of views.
// This could be used from vsip::subblock_domain.  However, first need
// to verify that pushing the view.block() into vsip::subblock_domain
// doesn't force it to always be called.

template <dimension_type D, typename B>
inline Domain<D>
block_subblock_domain(B const &block, Local_map const &, index_type sb)
{
  OVXX_PRECONDITION(sb == 0 || sb == no_subblock);
  return (sb == 0) ? block_domain<D>(block)
                   : empty_domain<D>();
}

template <dimension_type D, typename B, typename M>
inline Domain<D>
block_subblock_domain(B const &, M const &map, index_type sb)
{
  return map.template impl_subblock_domain<D>(sb);
}

} // namespace ovxx::parallel

/// Return the domain of BLOCK's subblock SB.
///
/// Requires
///   DIM to be dimension of block,
///   BLOCK to be a block,
///   SB to either be a valid subblock of BLOCK, or the value no_subblock.
///
/// Returns
///   The domain of BLOCK's subblock SB if SB is valid, the empty
///   domain if SB == no_subblock.
template <dimension_type D, typename B>
Domain<D> 
block_subblock_domain(B const &block, index_type sb)
{
  return parallel::block_subblock_domain<D, B>
    (block, block.map(), sb);
}

/// Return the domain of BLOCK's subblock held by the local processor.
///
/// Requires
///   BLOCK to be a view
///
/// Returns
///   The domain of BLOCK's subblock held by the local processor.
template <dimension_type D, typename B>
Domain<D>
block_subblock_domain(B const &block)
{
  return parallel::block_subblock_domain<D, B>
    (block, block.map(), block.map().subblock());
}

} // namespace ovxx

#endif

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
subblock_domain(B const &block, Local_map const &, index_type sb)
{
  OVXX_PRECONDITION(sb == 0 || sb == no_subblock);
  return (sb == 0) ? block_domain<D>(block) : empty_domain<D>();
}

template <dimension_type D, typename B, typename M>
inline Domain<D>
subblock_domain(B const &, M const &map, index_type sb)
{
  return map.template impl_subblock_domain<D>(sb);
}

/// Return the domain of BLOCK's subblock SB.
///
/// Requires
///   BLOCK to be a block,
///   SB to either be a valid subblock of BLOCK, or the value no_subblock.
///
/// Returns
///   The domain of BLOCK's subblock SB if SB is valid, the empty
///   domain if SB == no_subblock.
template <dimension_type D, typename B>
Domain<D>
subblock_domain(B const &block, index_type sb)
{
  return subblock_domain<D>(block, block.map(), sb);
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
subblock_domain(B const &block)
{
  return subblock_domain<D>(block, block.map(), block.map().subblock());
}

template <dimension_type D, typename B>
inline Domain<D>
local_domain(B const &block, Local_map const &, index_type sb, index_type p)
{
  OVXX_PRECONDITION((sb == 0 && p == 0) || sb == no_subblock);
  return (sb == 0) ? block_domain<D>(block) : empty_domain<D>();
}

template <dimension_type D, typename B, typename M>
inline Domain<D>
local_domain(B const &, M const &map, index_type sb, index_type p)
{
  return map.template impl_local_domain<D>(sb, p);
}

template <dimension_type D, typename B>
inline Domain<D>
local_domain(B const &block, index_type sb, index_type p)
{
  return local_domain<D>(block, block.map(), sb, p);
}

template <dimension_type D, typename B>
inline Domain<D>
local_domain(B const &block, index_type p = 0)
{
  return local_domain<D>(block, block.map(), block.map().subblock(), p);
}

template <dimension_type D, typename B>
inline Domain<D>
global_domain(B const &block, Local_map const &,
	      index_type sb, index_type p OVXX_UNUSED)
{
  OVXX_PRECONDITION((sb == 0 && p == 0) || sb == no_subblock);
  return sb == 0 ? block_domain<D>(block) : empty_domain<D>();
}

template <dimension_type D, typename B, typename M>
inline Domain<D>
global_domain(B const &, M const &map, index_type sb, index_type p)
{
  return map.template impl_global_domain<D>(sb, p);
}

template <dimension_type D, typename B>
inline Domain<D>
global_domain(B const &block, index_type sb, index_type p)
{
  return global_domain<D>(block, block.map(), sb, p);
}

template <dimension_type D, typename B>
inline Domain<D>
global_domain(B const &block, index_type p = 0)
{
  return global_domain<D>(block, block.map(), block.map().subblock(), p);
}

template <typename B>
length_type
block_num_patches(B const &block, index_type sb)
{
  return block.map().impl_num_patches(sb);
}

template <typename B>
length_type
block_num_patches(B const &block)
{
  return block.map().impl_num_patches(block.map().subblock());
}

template <typename B, dimension_type D>
index_type
block_subblock_from_global_index(B const &block, Index<D> const &idx)
{
  for (dimension_type d = 0; d < D; ++d)
    OVXX_PRECONDITION(idx[d] < block.size(D, d));

  return block.map().template impl_subblock_from_global_index<D>(idx);
}

template <typename B, dimension_type D>
index_type
block_patch_from_global_index(B const &block, Index<D> const &idx)
{
  for (dimension_type d = 0; d < D; ++d)
    OVXX_PRECONDITION(idx[d] < block.size(D, d));

  return block.map().template impl_subblock_from_global_index<D>(idx);
}

template <typename B>
index_type
block_local_from_global_index(B const &block, dimension_type dim, index_type idx)
  VSIP_NOTHROW
{
  return block.map().impl_local_from_global_index(dim, idx);
}

template <typename B, dimension_type D>
Index<D>
block_local_from_global_index(B const &block, Index<D> const &g_idx) VSIP_NOTHROW
{
  Index<D> l_idx;

  for (dimension_type d = 0; d < D; ++d)
    l_idx[d] = block_local_from_global_index(block, d, g_idx[d]);

  return l_idx;
}

template <typename B>
inline
index_type
block_global_from_local_index(B const &block, dimension_type dim,
			      index_type sb, index_type idx)
{
  return block.map().impl_global_from_local_index(dim, sb, idx);
}

template <typename B, dimension_type D>
inline
Index<D>
block_global_from_local_index(B const &block, index_type sb, Index<D> const &l_idx)
{
  Index<D> g_idx;

  for (dimension_type d = 0; d < D; ++d)
    g_idx[d] = block_global_from_local_index(block, d, sb, l_idx[d]);

  return g_idx;
}

template <typename B, dimension_type D>
inline
Index<D>
block_global_from_local_index(B const &block, Index<D> const &l_idx)
{
  Index<D> g_idx;

  for (dimension_type d=0; d<D; ++d)
    g_idx[d] = 
      block.map().impl_global_from_local_index(d, block.map().subblock(), l_idx[d]);

  return g_idx;
}

template <typename B>
inline
index_type
block_global_from_local_index(B const &block, dimension_type dim, index_type idx)
{
  index_type sb = block.map().subblock();

  if (sb != no_subblock)
    return block.map().impl_global_from_local_index(dim, sb, idx);
  else
    return no_index;
}

} // namespace ovxx::parallel
} // namespace ovxx

#endif

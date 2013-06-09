//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_parallel_hpp_
#define vsip_impl_parallel_hpp_

#include <vsip/support.hpp>
#include <vsip/impl/vector.hpp>
#include <ovxx/domain_utils.hpp>
#include <ovxx/parallel/support.hpp>

namespace vsip
{

inline const_Vector<processor_type> processor_set()
{
  static Dense<1, processor_type> *pset_block_ = 0;

  if (pset_block_ == 0)
  {
#if OVXX_PARALLEL
    namespace p = ovxx::parallel;
    p::Communicator::pvec_type const &pvec = p::default_communicator().pvec();
    pset_block_ = new Dense<1, processor_type>(Domain<1>(pvec.size()));
    for (index_type i=0; i<pvec.size(); ++i)
      pset_block_->put(i, pvec[i]);
#else
    pset_block_ = new Dense<1, processor_type>(1);
    pset_block_->put(0, 0);
#endif
  }

  return Vector<processor_type>(*pset_block_);
}

// [view.support.fcn] parallel support functions

/// Return the domain of VIEW's subblock SB.

/// Requires
///   VIEW to be a view
///   SB to either be a valid subblock of VIEW, or the value no_subblock.
///
/// Returns
///   The domain of VIEW's subblock SB if SB is valid, the empty
///   domain if SB == no_subblock.
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, Domain<V::dim> >::type
subblock_domain(V const &view, index_type sb)
{
  return ovxx::parallel::subblock_domain<V::dim>(view.block(), sb);
}

/// Return the domain of VIEW's subblock held by the local processor.

/// Requires
///   VIEW to be a view
///
/// Returns
///   The domain of VIEW's subblock held by the local processor.
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, Domain<V::dim> >::type
subblock_domain(V const &view)
{
  return ovxx::parallel::subblock_domain<V::dim>(view.block());
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
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, Domain<V::dim> >::type
local_domain(V const &view, index_type sb, index_type p)
{
  return ovxx::parallel::local_domain<V::dim>(view.block(), sb, p);
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
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, Domain<V::dim> >::type
local_domain(V const &view, index_type p=0)
{
  return ovxx::parallel::local_domain<V::dim>(view.block(), p);
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
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, Domain<V::dim> >::type
global_domain(V const &view, index_type sb, index_type p)
{
  return ovxx::parallel::global_domain<V::dim>(view.block(), sb, p);
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
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, Domain<V::dim> >::type
global_domain(V const &view, index_type p=0)
{
  return ovxx::parallel::global_domain<V::dim>(view.block(), p);
}

/// Return the number of subblocks VIEW is distrubted over.

/// Requires
///   VIEW to be a view
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, length_type>::type
num_subblocks(V const &view)
{
  return view.block().map().num_subblocks();
}

/// Return the number of patches in VIEW's subblock SB.

/// Requires
///   VIEW to be a view.
///   SB to either be a valid subblock of VIEW, or the value no_subblock.
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, length_type>::type
num_patches(V const &view, index_type sb)
{
  return ovxx::parallel::block_num_patches(view.block(), sb);
}

/// Return the number of patches in VIEW's subblock held on the local
/// processor.

/// Requires
///   VIEW to be a view.
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, length_type>::type
num_patches(V const &view)
{
  return ovxx::parallel::block_num_patches(view.block());
}



/// Return the subblock rank held by processor PR.

/// Requires
///   VIEW to be a view.
///   PR to be processor.
///
/// Returns
///   The subblock rank of VIEW held by processor PR if it holds a subblock,
///   NO_SUBBLOCK otherwise.
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, index_type>::type
subblock(V const &view, processor_type pr)
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
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, index_type>::type
subblock(V const &view)
{
  return view.block().map().subblock();
}

/// Determine which subblock holds VIEW's global index IDX
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, index_type>::type
subblock_from_global_index(V const &view, Index<V::dim> const &idx)
{
  return ovxx::parallel::block_subblock_from_global_index(view.block(), idx);
}

/// Determine which patch holds VIEW's global index IDX

/// Notes:
///   This patch is only valid in the subblock returned by
///   subblock_from_global_index.
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, index_type>::type
patch_from_global_index(V const &view, Index<V::dim> const &idx)
{
  return ovxx::parallel::block_patch_from_global_index(view.block(), idx);
}

/// Determine the local index corresponding to VIEW's global index G_IDX.

/// Notes:
///   This local index is only valid in processors hold the subblock
///   returned by subblock_from_global_index.
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, Index<V::dim> >::type
local_from_global_index(V const &view, Index<V::dim> const &idx) VSIP_NOTHROW
{
  return ovxx::parallel::block_local_from_global_index(view.block(), idx);
}

/// Determine the local index corresponding to VIEW's global index
/// G_IDX for dimension DIM.
template <typename V>
typename ovxx::enable_if<ovxx::is_view_type<V>::value, index_type>::type
local_from_global_index(V const &view, dimension_type dim, index_type idx)
  VSIP_NOTHROW
{
  return ovxx::parallel::block_local_from_global_index(view.block(), dim, idx);
}

/// Determine VIEW's global index corresponding to local index L_IDX
/// of subblock SB.
template <typename V>
inline
typename ovxx::enable_if<ovxx::is_view_type<V>::value, Index<V::dim> >::type
global_from_local_index(V const &view, index_type sb, Index<V::dim> const &idx)
{
  return ovxx::parallel::block_global_from_local_index(view.block(), sb, idx);
}

/// Determine VIEW's global index corresponding to local index L_IDX
/// of the subblock held on the local processor.
template <typename V>
inline
typename ovxx::enable_if<ovxx::is_view_type<V>::value, Index<V::dim> >::type
global_from_local_index(V const &view, Index<V::dim> const &idx)
{
  return ovxx::parallel::block_global_from_local_index(view.block(), idx);
}

/// Determine VIEW's global index corresponding to local index L_IDX
/// for dimension DIM of subblock SB.
template <typename V>
inline
typename ovxx::enable_if<ovxx::is_view_type<V>::value, index_type>::type
global_from_local_index(V const &view, dimension_type dim,
			index_type sb, index_type idx)
{
  return ovxx::parallel::block_global_from_local_index(view, dim, sb, idx);
}

/// Determine VIEW's global index corresponding to local index L_IDX
/// for dimension DIM of the subblock held on the local processor.
template <typename V>
inline
typename ovxx::enable_if<ovxx::is_view_type<V>::value, index_type>::type
global_from_local_index(V const &view, dimension_type dim, index_type idx)
{
  return ovxx::parallel::block_global_from_local_index(view.block(), dim, idx);
}

} // namespace vsip

#endif

//
// Copyright (c) 2005, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

/// Description
///   Forward declarations for Map and distributions.

#ifndef vsip_impl_map_fwd_hpp_
#define vsip_impl_map_fwd_hpp_

namespace vsip
{

// Forward Declarations

class Whole_dist;
class Block_dist;
class Cyclic_dist;

template <typename D0 = Block_dist,
	  typename D1 = Block_dist,
	  typename D2 = Block_dist>
class Map;

template <dimension_type D> class Replicated_map;

} // namespace vsip

namespace ovxx
{
namespace parallel
{
template <dimension_type D> class local_or_global_map;
template <dimension_type D> class scalar_map;
template <dimension_type D> class Subset_map;

template <dimension_type D, typename M> struct Map_project_1;
template <dimension_type D0, dimension_type D1, typename M> struct Map_project_2;
template <dimension_type D, typename M> struct Map_subdomain;

} // namespace ovxx::parallel
} // namespace ovxx

#endif

//
// Copyright (c) 2005, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

/// Description
///   Forward declarations for Map and distributions.

#ifndef VSIP_CORE_MAP_FWD_HPP
#define VSIP_CORE_MAP_FWD_HPP

namespace vsip
{

// Forward Declarations
class Whole_dist;

class Block_dist;

class Cyclic_dist;

template <typename Dim0 = Block_dist,
	  typename Dim1 = Block_dist,
	  typename Dim2 = Block_dist>
class Map;

template <dimension_type Dim>
class Global_map;

template <dimension_type Dim>
class Replicated_map;

template <dimension_type Dim>
class Local_or_global_map;

namespace impl
{

template <dimension_type Dim>
class Scalar_block_map;

template <dimension_type Dim>
class Subset_map;

template <dimension_type Dim,
	  typename       MapT>
struct Map_project_1;

template <dimension_type Dim0,
	  dimension_type Dim1,
	  typename       MapT>
struct Map_project_2;

template <dimension_type Dim,
	  typename       MapT>
struct Map_subdomain;

} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_MAP_FWD_HPP

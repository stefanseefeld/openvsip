//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_MAP_TRAITS_HPP
#define VSIP_CORE_MAP_TRAITS_HPP

/***********************************************************************
  Included Files
***********************************************************************/



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

/// Traits class to determine if a map is serial or not.

template <typename MapT>
struct Is_local_map
{ static bool const value = false; };

template <typename Map>
struct is_global_map
{ static bool const value = false; };

template <typename MapT>
struct Is_local_only
{ static bool const value = Is_local_map<MapT>::value &&
                           !is_global_map<MapT>::value; };

template <typename MapT>
struct Is_global_only
{ static bool const value = is_global_map<MapT>::value &&
                           !Is_local_map<MapT>::value; };

template <dimension_type Dim,
	  typename       MapT>
struct Is_block_dist
{ static bool const value = false; };


} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_MAP_TRAITS_HPP

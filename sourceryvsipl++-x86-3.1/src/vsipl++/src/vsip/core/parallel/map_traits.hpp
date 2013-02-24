/* Copyright (c) 2005, 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/map_traits.hpp
    @author  Jules Bergmann
    @date    2005-10-31
    @brief   VSIPL++ Library: Map traits.

    Traits for map types.
*/

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

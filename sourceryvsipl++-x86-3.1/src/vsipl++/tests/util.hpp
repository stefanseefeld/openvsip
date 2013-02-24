/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/util.hpp
    @author  Jules Bergmann
    @date    2005-05-10
    @brief   VSIPL++ Library: Test utilities.
*/

#ifndef VSIP_TESTS_UTIL_HPP
#define VSIP_TESTS_UTIL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

// Utility function to generalize creation of a view from a domain (1-dim).

template <typename View>
inline View
create_view(
  vsip::Domain<1> const&                     dom,
  typename View::block_type::map_type const& map =
		typename View::block_type::map_type())
{
  return View(dom[0].length(), map);
}



template <typename View>
inline View
create_view(
  vsip::Domain<1> const&                     dom,
  typename View::value_type                  val,
  typename View::block_type::map_type const& map =
		typename View::block_type::map_type())
{
  return View(dom[0].length(), val, map);
}



template <typename View>
inline View
create_view(
  vsip::Domain<1> const&                     dom,
  typename View::value_type*                 ptr,
  typename View::block_type::map_type const& map =
		typename View::block_type::map_type())
{
  typedef typename View::block_type block_type;

  block_type* block = new block_type(dom, ptr, map);
  View view(*block);
  block->decrement_count();
  return view;
}



// Utility function to generalize creation of a view from a domain (2-dim).

template <typename View>
inline View
create_view(
  vsip::Domain<2> const& dom,
  typename View::block_type::map_type const& map =
		typename View::block_type::map_type())
{
  return View(dom[0].length(), dom[1].length(), map);
}

template <typename View>
inline View
create_view(
  vsip::Domain<2> const&                     dom,
  typename View::value_type                  val,
  typename View::block_type::map_type const& map =
		typename View::block_type::map_type())
{
  return View(dom[0].length(), dom[1].length(), val, map);
}



// Utility function to generalize creation of a view from a domain (3-dim).

template <typename View>
inline View
create_view(
  vsip::Domain<3> const&                     dom,
  typename View::block_type::map_type const& map =
		typename View::block_type::map_type())
{
  return View(dom[0].length(), dom[1].length(), dom[2].length(), map);
}

template <typename View>
inline View
create_view(
  vsip::Domain<3> const&                     dom,
  typename View::value_type                  val,
  typename View::block_type::map_type const& map =
		typename View::block_type::map_type())
{
  return View(dom[0].length(), dom[1].length(), dom[2].length(), val, map);
}

#endif // VSIP_TESTS_UTIL_HPP

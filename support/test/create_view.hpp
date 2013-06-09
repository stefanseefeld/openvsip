//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef test_create_view_hpp_
#define test_create_view_hpp_

#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>

namespace test
{
// Utility function to generalize creation of a view from a domain (1-dim).
template <typename View>
inline View
create_view(vsip::Domain<1> const &dom,
	    typename View::block_type::map_type const &map =
	    typename View::block_type::map_type())
{
  return View(dom[0].length(), map);
}



template <typename View>
inline View
create_view(vsip::Domain<1> const &dom,
	    typename View::value_type val,
	    typename View::block_type::map_type const &map =
	    typename View::block_type::map_type())
{
  return View(dom[0].length(), val, map);
}



template <typename View>
inline View
create_view(vsip::Domain<1> const &dom,
	    typename View::value_type *ptr,
	    typename View::block_type::map_type const &map =
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
create_view(vsip::Domain<2> const &dom,
	    typename View::block_type::map_type const &map =
	    typename View::block_type::map_type())
{
  return View(dom[0].length(), dom[1].length(), map);
}

template <typename View>
inline View
create_view(vsip::Domain<2> const &dom,
	    typename View::value_type val,
	    typename View::block_type::map_type const &map =
	    typename View::block_type::map_type())
{
  return View(dom[0].length(), dom[1].length(), val, map);
}



// Utility function to generalize creation of a view from a domain (3-dim).

template <typename View>
inline View
create_view(vsip::Domain<3> const &dom,
	    typename View::block_type::map_type const &map =
	    typename View::block_type::map_type())
{
  return View(dom[0].length(), dom[1].length(), dom[2].length(), map);
}

template <typename View>
inline View
create_view(vsip::Domain<3> const &dom,
	    typename View::value_type val,
	    typename View::block_type::map_type const &map =
	    typename View::block_type::map_type())
{
  return View(dom[0].length(), dom[1].length(), dom[2].length(), val, map);
}
} // namespace test

#endif

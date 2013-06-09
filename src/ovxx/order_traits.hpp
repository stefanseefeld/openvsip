//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_order_traits_hpp_
#define ovxx_order_traits_hpp_

#include <ovxx/support.hpp>

namespace ovxx
{

template <typename O, dimension_type D>
struct dim_of;

template <dimension_type D0,
	  dimension_type D1,
	  dimension_type D2>
struct dim_of<tuple<D0, D1, D2>, 0>
{ static dimension_type const value = D0;};

template <dimension_type D0,
	  dimension_type D1,
	  dimension_type D2>
struct dim_of<tuple<D0, D1, D2>, 1>
{ static dimension_type const value = D1;};

template <dimension_type D0,
	  dimension_type D1,
	  dimension_type D2>
struct dim_of<tuple<D0, D1, D2>, 2>
{ static dimension_type const value = D2;};

// is_order_valid is a metafunction that returns true if the given
// O is valid for the given dimension D.
template <dimension_type D, typename O>
struct is_order_valid { static bool const value = false;};

template <>
struct is_order_valid<1, row1_type> { static bool const value = true;};
template <>
struct is_order_valid<2, row2_type> { static bool const value = true;};
template <>
struct is_order_valid<2, col2_type> { static bool const value = true;};
template <>
struct is_order_valid<3, tuple<0,1,2> > { static bool const value = true;};
template <>
struct is_order_valid<3, tuple<0,2,1> > { static bool const value = true;};
template <>
struct is_order_valid<3, tuple<1,0,2> > { static bool const value = true;};
template <>
struct is_order_valid<3, tuple<1,2,0> > { static bool const value = true;};
template <>
struct is_order_valid<3, tuple<2,0,1> > { static bool const value = true;};
template <>
struct is_order_valid<3, tuple<2,1,0> > { static bool const value = true;};

} // namespace ovxx

#endif

//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_ORDER_TRAITS_HPP
#define VSIP_CORE_ORDER_TRAITS_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

// Define convenience trait Dim_of.  It extracts the n'th dimension
// from a dimension order tuple.

template <typename       OrderT,
	  dimension_type Dim>
struct Dim_of;

template <dimension_type Dim0,
	  dimension_type Dim1,
	  dimension_type Dim2>
struct Dim_of<tuple<Dim0, Dim1, Dim2>, 0>
{ static dimension_type const value = Dim0; };

template <dimension_type Dim0,
	  dimension_type Dim1,
	  dimension_type Dim2>
struct Dim_of<tuple<Dim0, Dim1, Dim2>, 1>
{ static dimension_type const value = Dim1; };

template <dimension_type Dim0,
	  dimension_type Dim1,
	  dimension_type Dim2>
struct Dim_of<tuple<Dim0, Dim1, Dim2>, 2>
{ static dimension_type const value = Dim2; };

// Is_order_valid is a metafunction that returns true if the given
// OrderT is valid for the given dimension D.
template <dimension_type D, typename OrderT>
struct Is_order_valid { static bool const value = false;};

template <>
struct Is_order_valid<1, row1_type> { static bool const value = true;};
template <>
struct Is_order_valid<2, row2_type> { static bool const value = true;};
template <>
struct Is_order_valid<2, col2_type> { static bool const value = true;};
template <>
struct Is_order_valid<3, tuple<0,1,2> > { static bool const value = true;};
template <>
struct Is_order_valid<3, tuple<0,2,1> > { static bool const value = true;};
template <>
struct Is_order_valid<3, tuple<1,0,2> > { static bool const value = true;};
template <>
struct Is_order_valid<3, tuple<1,2,0> > { static bool const value = true;};
template <>
struct Is_order_valid<3, tuple<2,0,1> > { static bool const value = true;};
template <>
struct Is_order_valid<3, tuple<2,1,0> > { static bool const value = true;};


} // namespace impl

} // namespace vsip

#endif

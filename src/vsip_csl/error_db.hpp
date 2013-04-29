/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/error_db.cpp
    @author  Jules Bergmann
    @date    2005-12-12
    @brief   VSIPL++ CodeSourcery Library: Measure difference between 
             views in decibels.
*/

#ifndef VSIP_CSL_ERROR_DB_HPP
#define VSIP_CSL_ERROR_DB_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <algorithm>

#include <vsip/math.hpp>
#include <vsip/core/view_cast.hpp>
#include <vsip_csl/test.hpp>


namespace vsip_csl
{

/***********************************************************************
  Definitions
***********************************************************************/

namespace impl
{

template <typename T>
struct Error_db_traits
{
  typedef T type;
};

template <> struct Error_db_traits<unsigned char>  { typedef float type; };
template <> struct Error_db_traits<unsigned short> { typedef float type; };
template <> struct Error_db_traits<unsigned int>   { typedef float type; };
template <> struct Error_db_traits<signed char>  { typedef float type; };
template <> struct Error_db_traits<signed short> { typedef float type; };
template <> struct Error_db_traits<signed int>   { typedef float type; };

template <typename T>
struct Error_db_traits<std::complex<T> >
{
  typedef std::complex<typename Error_db_traits<T>::type> type;
};

} // namespace vsip_csl::impl



// Compute the distance between two views in terms of relative magsq
// difference in decibels.
//
// Requires
//   V1 and V2 to be VSIPL++ views of the same dimensionality
//      and same value type.
//
// Returns
//   The result is computed by the equation:
//
//                            max(magsq(v1 - v2))
//     error_db = 10 * log10( ------------------- )
//                               2 * refmax
//
//   Smaller (more negative) error dBs are better.  An error dB of -201
//   indicates that the two views are practically equal.  An error dB of
//   0 indicates that the views have elements that are negated
//   (v1(idx) == -v2(idx)).
//
//   If either input contains a NaN value, an error dB of 201 is returned.
//
//   For example, an error dB of -50 indicates that
//
//       max(magsq(v1 - v2)
//     ( ------------------ ) < 10^-5
//          2 * refmax

template <template <typename, typename> class View1,
	  template <typename, typename> class View2,
	  typename                            T1,
	  typename                            T2,
	  typename                            Block1,
	  typename                            Block2>
inline double
error_db(
  View1<T1, Block1> v1,
  View2<T2, Block2> v2)
{
  using vsip::impl::Dim_of_view;
  using vsip::dimension_type;
  using vsip::impl::view_cast;
  using vsip_csl::impl::Error_db_traits;

  typedef typename Error_db_traits<T1>::type promote1_type;
  typedef typename Error_db_traits<T2>::type promote2_type;
  // typedef typename vsip::Promotion<diff1_type, diff2_type>::type diff_type;

  // garbage in, garbage out.
  if (anytrue(is_nan(v1)) || anytrue(is_nan(v2)))
    return 201.0;

  dimension_type const dim = Dim_of_view<View2>::dim;

  vsip::Index<dim> idx;

  typename vsip::impl::View_cast<promote1_type, View1, T1, Block1>::view_type
    pv1 = view_cast<promote1_type>(v1);
  typename vsip::impl::View_cast<promote2_type, View2, T2, Block2>::view_type
    pv2 = view_cast<promote2_type>(v2);

  double refmax1 = maxval(magsq(pv1), idx);
  double refmax2 = maxval(magsq(pv2), idx);
  double refmax  = std::max(refmax1, refmax2);
  double maxsum  = maxval(
    ite(magsq((pv1 - pv2)) < 1.e-20,
	-201.0,
	10.0 * log10(magsq((pv1 - pv2))/(2.0*refmax)) ),
    idx);
  return maxsum;
}

} // namespace vsip_csl

#endif // VSIP_CSL_ERROR_DB_HPP

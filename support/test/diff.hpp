//
// Copyright (c) 2005, 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef test_diff_hpp_
#define test_diff_hpp_

#include <vsip/math.hpp>
#include <ovxx/view/cast.hpp>
#include <algorithm>

namespace test
{
using namespace ovxx;
namespace detail
{

template <typename T>
struct diff_traits { typedef T type;};

template <> struct diff_traits<unsigned char>  { typedef float type;};
template <> struct diff_traits<unsigned short> { typedef float type;};
template <> struct diff_traits<unsigned int>   { typedef float type;};
template <> struct diff_traits<signed char>  { typedef float type;};
template <> struct diff_traits<signed short> { typedef float type;};
template <> struct diff_traits<signed int>   { typedef float type;};

template <typename T>
struct diff_traits<std::complex<T> >
{
  typedef complex<typename diff_traits<T>::type> type;
};

} // namespace ovxx::test::detail


// Compute the distance between two views in terms of relative magsq
// difference in decibels.
//
// The result is computed by the equation:
//
//                          max(magsq(v1 - v2))
//     result = 10 * log10( ------------------- )
//                             2 * refmax
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
//      max(magsq(v1 - v2))
//      ------------------  < 10^-5
//         2 * refmax
template <template <typename, typename> class V1,
	  typename T1, typename B1,
	  template <typename, typename> class V2,
	  typename T2, typename B2>
inline double
diff(V1<T1, B1> v1, V2<T2, B2> v2)
{
  using detail::diff_traits;

  typedef typename diff_traits<T1>::type p1_type;
  typedef typename diff_traits<T2>::type p2_type;

  // garbage in, garbage out.
  if (anytrue(is_nan(v1)) || anytrue(is_nan(v2)))
    return 201.0;

  dimension_type const dim = dim_of_view<V2>::dim;

  Index<dim> idx;

  typename ovxx::detail::view_cast<p1_type, V1, T1, B1>::view_type
    pv1 = view_cast<p1_type>(v1);
  typename ovxx::detail::view_cast<p2_type, V2, T2, B2>::view_type
    pv2 = view_cast<p2_type>(v2);

  double refmax1 = maxval(magsq(pv1), idx);
  double refmax2 = maxval(magsq(pv2), idx);
  double refmax  = std::max(refmax1, refmax2);
  double maxsum  = 
    maxval(ite(magsq((pv1 - pv2)) < 1.e-20,
	       -201.0,
	       10.0 * log10(magsq((pv1 - pv2))/(2.0*refmax))),
	   idx);
  return maxsum;
}

} // namespace test

#endif

//
// Copyright (c) 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_equal_hpp_
#define ovxx_equal_hpp_

#include <ovxx/support.hpp>
#include <ovxx/complex_traits.hpp>
#include <ovxx/math/scalar.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>

namespace ovxx
{

// This code was inspired by
// http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
template <typename T>
bool almost_equal(T a, T b, 
		  typename scalar_of<T>::type rel_eps = 1e-4,
		  typename scalar_of<T>::type abs_eps = 1e-6)
{
  if (math::mag(a - b) < abs_eps) return true;
  typename scalar_of<T>::type relative_error;
   
  if (math::mag(a) > math::mag(b))
    relative_error = math::mag((a - b) / b);
  else
    relative_error = math::mag((b - a) / a);
    
  return relative_error <= rel_eps;
}

template <typename T1, typename T2>
inline typename enable_if<is_integral<T1>::value &&
                          is_integral<T2>::value,
			  bool>::type
equal(T1 a, T2 b)
{ return a == b;}

template <typename T1, typename T2>
inline typename enable_if<is_floating_point<T1>::value ||
                          is_floating_point<T2>::value,
			  bool>::type
equal(T1 a, T2 b)
{
  return almost_equal<typename Promotion<T1, T2>::type>(a, b);
}

template <typename T1, typename T2>
inline bool equal(complex<T1> const &a, complex<T2> const &b)
{ return equal(a.real(), b.real()) && equal(a.imag(), b.imag());}

template <typename B, dimension_type D>
inline bool equal(element_proxy<B, D> const &a, typename B::value_type b)
{
  return equal(static_cast<typename B::value_type>(a), b);
}

template <typename B, dimension_type D>
inline bool equal(typename B::value_type a, element_proxy<B, D> const &b) 
{
  return equal(a, static_cast<typename B::value_type>(b));
}

template <typename B1, dimension_type D1, typename B2, dimension_type D2>
inline bool equal(element_proxy<B1, D1> const &a, element_proxy<B2, D2> const &b)
{
  return equal(static_cast<typename B1::value_type>(a), static_cast<typename B2::value_type>(b));
}

template <typename B, dimension_type D>
inline bool equal(element_proxy<B, D> const &a, element_proxy<B, D> const &b)
{
  return equal(static_cast<typename B::value_type>(a), static_cast<typename B::value_type>(b));
}

template <typename T1, typename B1, typename T2, typename B2>
inline bool equal(const_Vector<T1, B1> v, const_Vector<T2, B2> w)
{
  if (v.size() != w.size()) return false;
  for (length_type i = 0; i != v.size(); ++i)
    if (!equal(v.get(i), w.get(i)))
      return false;
  return true;
}

template <typename T1, typename B1, typename T2, typename B2>
inline bool equal(const_Matrix<T1, B1> v, const_Matrix<T2, B2> w)
{
  if (v.size(0) != w.size(0) || v.size(1) != w.size(1)) return false;
  for (length_type i = 0; i != v.size(0); ++i)
    for (length_type j = 0; j != v.size(1); ++j)
      if (!equal(v.get(i, j), w.get(i, j)))
	return false;
  return true;
}

template <typename T1, typename B1, typename T2, typename B2>
inline bool equal(const_Tensor<T1, B1> v, const_Tensor<T2, B2> w)
{
  if (v.size(0) != w.size(0) || v.size(1) != w.size(1) || v.size(2) != w.size(2))
    return false;
  for (length_type i = 0; i != v.size(0); ++i)
    for (length_type j = 0; j != v.size(1); ++j)
      for (length_type k = 0; k != v.size(2); ++k)
	if (!equal(v.get(i, j, k), w.get(i, j, k)))
	  return false;
  return true;
}
} // namespace ovxx

#endif

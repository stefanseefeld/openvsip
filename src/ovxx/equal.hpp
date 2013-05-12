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
  T relative_error;
   
  if (math::mag(a) > math::mag(b))
    relative_error = math::mag((a - b) / b);
  else
    relative_error = math::mag((b - a) / a);
    
  return relative_error <= rel_eps;
}

template <typename T>
inline typename enable_if<is_integral<T>::value, bool>::type
equal(T a, T b)
{ return a == b;}

template <typename T>
inline typename enable_if<is_floating_point<T>::value, bool>::type
equal(T a, T b)
{ return almost_equal<T>(a, b);}

template <typename T1, typename T2>
inline bool equal(complex<T1> const &a, complex<T2> const &b)
{ return equal(a.real(), b.real()) && equal(a.imag(), b.imag());}

} // namespace ovxx

#endif

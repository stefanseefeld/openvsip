//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_complex_traits_hpp_
#define ovxx_complex_traits_hpp_

#include <ovxx/c++11.hpp>
#include <complex>

namespace ovxx
{
using std::complex;

template <typename T> struct scalar_of 
{ typedef T type;};

template <typename T> struct scalar_of<complex<T> >
{ typedef typename scalar_of<T>::type type;};

template <typename T>
struct complex_of { typedef complex<T> type;};

template <typename T>
struct complex_of<complex<T> > { typedef complex<T> type;};

template <typename T>
struct is_complex { static bool const value = false;};

template <typename T>
struct is_complex<complex<T> > { static bool const value = true;};

namespace detail
{
// Convenience construct to generate a scalar-type
// that is equal to scalar_of<T>::type iff T is complex,
// and a dummy type otherwise.
template <typename T, typename Dummy>
struct complex_value_type
{
  typedef Dummy type; 
};

template <typename T, typename Dummy>
struct complex_value_type<complex<T>, Dummy>
{
  typedef T type;
};

template <typename T>
struct complex_traits
{
  static T conj(T val) { return val;}
  static T real(T val) { return val;}
  static T imag(T val) { return T();}
};

template <typename T>
struct complex_traits<std::complex<T> >
{
  static std::complex<T> conj(std::complex<T> val) { return std::conj(val);}
  static T real(std::complex<T> val) { return val.real();}
  static T imag(std::complex<T> val) { return val.imag();}
};

} // namespace ovxx::detail
} // namespace ovxx

#endif

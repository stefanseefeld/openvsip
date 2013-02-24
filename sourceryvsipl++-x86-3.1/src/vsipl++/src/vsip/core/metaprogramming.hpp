/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/metaprogramming.hpp
    @author  Jules Bergmann
    @date    2005-04-11
    @brief   VSIPL++ Library: Utilities for meta-programming with templates.
*/

#ifndef VSIP_CORE_METAPROGRAMMING_HPP
#define VSIP_CORE_METAPROGRAMMING_HPP

#include <vsip/core/c++0x.hpp>
#include <complex>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

/// Compare a compile-time value against a run-time value.

/// Useful for avoiding '-W -Wall' warnings when comparing a compile-time
/// template parameters with a run-time variable.

template <typename T, T X>
struct Compare
{
  bool operator>(T t) { return X > t; }
};


/// Speclialization for T == unsigned and X == 0.  This avoids a GCC
/// '-W -Wall' warning that comparison against compile-time 0 is
/// always true.

template <>
struct Compare<unsigned, 0>
{
  bool operator>(unsigned /*t*/) { return false; }
};

template <typename T> struct scalar_of 
  { typedef T type; };
template <typename T> struct scalar_of<std::complex<T> >
  { typedef typename scalar_of<T>::type type; };

template <typename T>
struct complex_of 
{ typedef std::complex<T> type; };

template <typename T>
struct complex_of<std::complex<T> >
{ typedef std::complex<T> type; };

template <typename T>
struct is_complex
{ static bool const value = false; };

template <typename T>
struct is_complex<std::complex<T> >
{ static bool const value = true; };


template <int Value>
struct Int_type
{ static const int value = Value; };

// Strip const qualifier from type.

template <typename T>
struct Non_const_of
{ typedef T type; };

template <typename T>
struct Non_const_of<T const>
{ typedef T type; };



// Compare two pointers for equality

template <typename Ptr1,
	  typename Ptr2>
struct Is_same_ptr
{
  static bool compare(Ptr1, Ptr2) { return false; }
};

template <typename PtrT>
struct Is_same_ptr<PtrT, PtrT>
{
  static bool compare(PtrT ptr1, PtrT ptr2) { return ptr1 == ptr2; }
};

template <typename Ptr1, typename Ptr2>
inline bool
is_same_ptr(Ptr1* ptr1, Ptr2* ptr2)
{
  typedef typename add_const<Ptr1>::type c_ptr1_type;
  typedef typename add_const<Ptr2>::type c_ptr2_type;
  return Is_same_ptr<c_ptr1_type*, c_ptr2_type*>::compare(ptr1, ptr2);
}

template <typename Ptr1,
	  typename Ptr2>
inline bool
is_same_ptr(std::pair<Ptr1*, Ptr1*> const& ptr1,
	    std::pair<Ptr2*, Ptr2*> const& ptr2)
{
  return is_same_ptr(ptr1.first, ptr2.first) &&
    is_same_ptr(ptr1.second, ptr2.second);
}


namespace detail
{

typedef char (&no_tag)[1];
typedef char (&yes_tag)[2];

template <typename T> no_tag has_type(...);
template <typename T> yes_tag has_type(int, typename T::type * = 0);

} // namespace detail

/// Determine whether a given type T has a 'type' member type.
template <typename T>
struct has_type
{
  static bool const value = 
    sizeof(detail::has_type<T>(0)) == sizeof(detail::yes_tag);
};

/// Lazy variant of `enable_if`, for cases where `T::type` may not be instantiable
/// if the condition `B` evaluates to `false`.
/// See http://www.boost.org/doc/libs/1_44_0/libs/utility/enable_if.html
/// for a discussion of the technique.
template <bool B, typename T>
struct lazy_enable_if_c { typedef typename T::type type;};

template <typename T>
struct lazy_enable_if_c<false, T> {};

template <typename C, typename T>
struct lazy_enable_if : public lazy_enable_if_c<C::value, T> {};

} // namespace impl
} // namespace vsip

#endif // VSIP_CORE_METAPROGRAMMING_HPP

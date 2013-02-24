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

/// If-Then-Else class.  Chooses either IfType::type or ElseType::type
/// based on boolean predicate.

template <bool     Predicate,
	  typename IfType,
	  typename ElseType>
struct ITE_Type;

template <typename IfType,
	  typename ElseType>
struct ITE_Type<true, IfType, ElseType>
{
  typedef typename IfType::type type;
};

template <typename IfType,
	  typename ElseType>
struct ITE_Type<false, IfType, ElseType>
{
  typedef typename ElseType::type type;
};



/// Wrap a type so that it can be accessed via ::type.

template <typename T>
struct As_type
{
  typedef T type;
};



/// Compare two types for "equality".

template <typename T1,
	  typename T2>
struct Type_equal
{
  static bool const value = false;
};

template <typename T>
struct Type_equal<T, T>
{
  static bool const value = true;
  typedef T type;
};



/// Pass type through if boolean value is true.  Used for SFINAE.

template <typename T,
	  bool     Bool>
struct Type_if;

template <typename T>
struct Type_if<T, true>
{
  typedef T type;
};



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

template <typename T> struct Scalar_of 
  { typedef T type; };
template <typename T> struct Scalar_of<std::complex<T> >
  { typedef typename Scalar_of<T>::type type; };

template <typename T>
struct Complex_of 
{ typedef std::complex<T> type; };

template <typename T>
struct Complex_of<std::complex<T> >
{ typedef std::complex<T> type; };

template <typename T>
struct Is_complex
{ static bool const value = false; };

template <typename T>
struct Is_complex<std::complex<T> >
{ static bool const value = true; };


template <bool Value>
struct Bool_type
{ static const bool value = Value; };

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

template <typename Ptr1,
	  typename Ptr2>
inline bool
is_same_ptr(
  Ptr1* ptr1,
  Ptr2* ptr2)
{
  return Is_same_ptr<Ptr1*, Ptr2*>::compare(ptr1, ptr2);
}

template <typename Ptr1,
	  typename Ptr2>
inline bool
is_same_ptr(
  std::pair<Ptr1*, Ptr1*> const& ptr1,
  std::pair<Ptr2*, Ptr2*> const& ptr2)
{
  return Is_same_ptr<Ptr1*, Ptr2*>::compare(ptr1.first, ptr2.first) &&
         Is_same_ptr<Ptr1*, Ptr2*>::compare(ptr1.second, ptr2.second);
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

} // namespace impl
} // namespace vsip

#endif // VSIP_CORE_METAPROGRAMMING_HPP

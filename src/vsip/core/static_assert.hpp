/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/static_assert.hpp
    @author  Jules Bergmann
    @date    2005-02-08
    @brief   VSIPL++ Library: compile time property checks.
*/

#ifndef VSIP_CORE_STATIC_ASSERT_HPP
#define VSIP_CORE_STATIC_ASSERT_HPP



/***********************************************************************
  Macros
***********************************************************************/

#define VSIP_IMPL_STATIC_ASSERT(expr)			\
  vsip::impl::Compile_time_assert<expr>::test();



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{


/// Compile_time_assert is a base class for creating compile-time checks.
///
/// It takes a single boolean template parameter, and works by only
/// providing a specialization for true.  Attempting to instantiate a
/// Compile_time_assert<false> or call Compile_time_assert<false>::test()
/// causes a compile-time error.
///
/// It can be used two ways:
///  - First, a concept check class can derive from Compile_time_assert,
///    passing a boolean expression to template parameter B.
///    For an example, see Assert_unsigned below.
///
///  - Second, code can form a compile-time assertion that EXPR is
///    true the macro VSIP_IMPL_STATIC_ASSERT, which in turn tries
///    to call Compile_time_assert<EXPR>::test(). If EXPR is true,
///    this results in a no-op.  If EXPR is is false, this results
///    in a compilation error.
///
template <bool B>
struct Compile_time_assert;


// specialize for true
template <>
struct Compile_time_assert<true>
{
  static void test() {}
};

// no specialization for false -- this triggers compile-time error.



/// Compile_time_assert_msg
template <bool B,
	  typename MsgT>
struct Compile_time_assert_msg;

template <typename MsgT>
struct Compile_time_assert_msg<true, MsgT>
{
  static void test() {}
};



/// Assert_unsigned<T> checks that type T is an unsigned at compile-time.
template<typename T> 
struct Is_unsigned
{
  static bool const value = (0 < T(-1));
};

template<typename T> 
struct Is_const
{
  static bool const value = false;
};
template<typename T> 
struct Is_const<T const>
{
  static bool const value = true;
};

template <typename T>
struct Assert_unsigned : Compile_time_assert<Is_unsigned<T>::value>
{};

template <typename T> struct Assert_const;
template <typename T> struct Assert_const<T const> {};

template <typename T> struct Assert_non_const {};
template <typename T> struct Assert_non_const<T const>;

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_STATIC_ASSERT_HPP

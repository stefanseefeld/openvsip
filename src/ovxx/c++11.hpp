//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cxx11_hpp_
#define ovxx_cxx11_hpp_

#include <ovxx/config.hpp>
#include <ovxx/c++11/shared_ptr.hpp>

namespace ovxx
{
namespace cxx11
{

#ifdef __GXX_EXPERIMENTAL_CXX0X__

using std::integral_constant;
using std::true_type;
using std::false_type;
using std::enable_if;
using std::conditional;
using std::is_same;
using std::is_const;
using std::is_volatile;
using std::remove_const;
using std::remove_volatile;
using std::remove_cv;
using std::add_const;
using std::add_volatile;
using std::add_cv;
using std::is_integral;
using std::is_floating_point;
using std::is_unsigned;
using std::is_arithmetic;

#else

template <typename T, T v>
struct integral_constant
{
  static const T value = v;
  typedef T value_type;
  typedef integral_constant<T, v> type;
};

template <typename T, T v>
const T integral_constant<T, v>::value;

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

template <bool B, typename T = void>
struct enable_if { typedef T type;};

template <typename T>
struct enable_if<false, T> {};

/// A conditional expression, but for types. If true, first, if false, second.
template<bool B, typename Iftrue, typename Iffalse>
struct conditional
{
  typedef Iftrue type;
};

template<typename Iftrue, typename Iffalse>
struct conditional<false, Iftrue, Iffalse>
{
  typedef Iffalse type;
};

template <typename T1, typename T2> struct is_same : false_type {};
template <typename T> struct is_same<T, T> : true_type {};

template <typename> struct is_const : public false_type {};
template <typename T> struct is_const<T const> : public true_type {};

template <typename> struct is_volatile : public false_type {};
template <typename T> struct is_volatile<T volatile> : public true_type {};

template <typename T> struct remove_const { typedef T type;};
template <typename T> struct remove_const<T const> { typedef T type;};

template <typename T> struct remove_volatile { typedef T type;};
template <typename T> struct remove_volatile<T volatile> { typedef T type;};

template <typename T> struct remove_cv 
{
  typedef typename remove_const<typename remove_volatile<T>::type>::type type;
};
  
template <typename T> struct add_const { typedef T const type;};
template <typename T> struct add_volatile { typedef T volatile type;};
template <typename T> struct add_cv 
{
  typedef typename
  add_const<typename add_volatile<T>::type>::type type;
};

template <typename T> struct is_integral : false_type {};
template <> struct is_integral<bool> : true_type {};
template <> struct is_integral<char> : true_type {};
template <> struct is_integral<signed char> : true_type {};
template <> struct is_integral<unsigned char> : true_type {};
template <> struct is_integral<wchar_t> : true_type {};
template <> struct is_integral<short> : true_type {};
template <> struct is_integral<unsigned short> : true_type {};
template <> struct is_integral<int> : true_type {};
template <> struct is_integral<unsigned int> : true_type {};
template <> struct is_integral<long> : true_type {};
template <> struct is_integral<unsigned long> : true_type {};
template <> struct is_integral<long long> : true_type {};
template <> struct is_integral<unsigned long long> : true_type {};

template <typename T> struct is_floating_point { static bool const value = false;};
template <> struct is_floating_point<float> { static bool const value = true;};
template <> struct is_floating_point<double> { static bool const value = true;};
template <> struct is_floating_point<long double> { static bool const value = true;};

template <typename T> struct is_unsigned { static bool const value = (0 < T(-1));};

template <typename T> 
struct is_arithmetic : integral_constant<bool,
					 is_integral<T>::value ||
					 is_floating_point<T>::value> {};
#endif
} // namespace ovxx::cxx11

using namespace cxx11;

#if OVXX_ENABLE_THREADING
# if __GNUC__
#  define thread_local __thread
# else
#  error "No support for threading with this compiler."
# endif
#else
# define thread_local
#endif

} // namespace ovxx

#endif

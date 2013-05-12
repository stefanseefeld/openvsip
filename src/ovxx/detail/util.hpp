//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_detail_util_hpp_
#define ovxx_detail_util_hpp_

namespace ovxx
{
namespace detail
{

typedef char (&no_tag)[1];
typedef char (&yes_tag)[2];

// Compare a compile-time value against a run-time value.
// Useful for avoiding '-W -Wall' warnings when comparing a compile-time
// template parameters with a run-time variable.
template <typename T, T X>
struct compare
{
  bool operator>(T t) { return X > t;}
};

// Specialization for T == unsigned and X == 0.  This avoids a GCC
// '-W -Wall' warning that comparison against compile-time 0 is
// always true.
template <>
struct compare<unsigned, 0>
{
  bool operator>(unsigned /*t*/) { return false;}
};

template <typename A, typename B>
struct is_same
{
  static bool compare(A const&, B const&) { return false;}
};

template <typename T>
struct is_same<T, T>
{
  static bool compare(T const &a, T const &b) { return &a == &b;}
};

} // namespace ovxx::detail
} // namespace ovxx

#endif

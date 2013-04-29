//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_ct_assert_hpp_
#define ovxx_ct_assert_hpp_

#define OVXX_CT_ASSERT(expr) ovxx::ct_assert<expr>::test();

namespace ovxx
{

/// ct_assert is a base class for creating compile-time checks.
///
/// It takes a single boolean template parameter, and works by only
/// providing a specialization for true.  Attempting to instantiate a
/// ct_assert<false> or call ct_assert<false>::test()
/// causes a compile-time error.
///
/// It can be used two ways:
///  - First, a concept check class can derive from ct_assert,
///    passing a boolean expression to template parameter B.
///    For an example, see Assert_unsigned below.
///
///  - Second, code can form a compile-time assertion that EXPR is
///    true the macro OVXX_CT_ASSERT, which in turn tries
///    to call ct_assert<EXPR>::test(). If EXPR is true,
///    this results in a no-op.  If EXPR is is false, this results
///    in a compilation error.
///
template <bool B>
struct ct_assert;

template <>
struct ct_assert<true>
{
  static void test() {}
};

// no specialization for false -- this triggers compile-time error.

template <bool B, typename MsgT>
struct ct_assert_msg;

template <typename MsgT>
struct ct_assert_msg<true, MsgT>
{
  static void test() {}
};

} // namespace ovxx

#endif

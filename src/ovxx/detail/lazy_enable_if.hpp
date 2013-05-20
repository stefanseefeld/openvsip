//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_detail_lazy_enable_if_hpp_
#define ovxx_detail_lazy_enable_if_hpp_

namespace ovxx
{
namespace detail
{
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

} // namespace ovxx::detail
} // namespace ovxx

#endif

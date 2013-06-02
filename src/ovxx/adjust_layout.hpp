//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_adjust_layout_hpp_
#define ovxx_adjust_layout_hpp_

#include <ovxx/layout.hpp>

namespace ovxx
{

/// Adjust a type:
///   if R == T || R == any_type:
///     type = T
///     compatible = true
///   else
///     type = R
///     compatible = false
template <typename R, typename T>
struct adjust_type
{
  typedef R type;
  static bool const compatible = false;
};

template <typename T>
struct adjust_type<any_type, T>
{
  typedef T type;
  static bool const compatible = true;
};

template <typename T>
struct adjust_type<T, T>
{
  typedef T type;
  static bool const compatible = true;
};

/// Adjust actual packing to requested packing:
template <pack_type R, pack_type P>
struct adjust_packing
{
  static pack_type const value = 
    R == any_packing ? (P == any_packing ? dense : P) : R;
  static bool const compatible =
    R == P || R == any_packing ||
    (R == unit_stride && is_packing_unit_stride<P>::value);
};

template <storage_format_type R, storage_format_type S>
struct adjust_storage_format
{
  static storage_format_type const value =
    R == any_storage_format ? (S == any_storage_format ? array : S) : R;
  static bool const compatible = 
    R == S ||
    R == any_storage_format ||
    // array and interleaved_complex are cast-compatible
    (R != split_complex && S != split_complex);
};

/// Adjust an actual layout to a requested layout.
///
/// Template parameters:
///   R: requested layout
///   L: actual layout
template <typename R, typename L>
struct adjust_layout
{
  static dimension_type const dim = R::dim;

  typedef typename R::order_type req_order_type;
  static pack_type const req_packing = R::packing;
  static storage_format_type const req_storage_format = R::storage_format;

  typedef typename L::order_type act_order_type;
  static pack_type const act_packing = L::packing;
  static storage_format_type const act_storage_format = L::storage_format;

  typedef typename adjust_type<req_order_type, act_order_type>::type order_type;
  static pack_type const packing = adjust_packing<req_packing, act_packing>::value;
  static storage_format_type const storage_format = 
    adjust_storage_format<req_storage_format, act_storage_format>::value;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <dimension_type D, typename L>
struct adjust_layout_dim
{
  typedef Layout<D, typename L::order_type, L::packing, L::storage_format> type;
};

template <pack_type P, typename L>
struct adjust_layout_packing
{
  typedef Layout<L::dim, typename L::order_type, P, L::storage_format> type;
};

template <storage_format_type S, typename L>
struct adjust_layout_storage_format
{
  typedef Layout<L::dim, typename L::order_type, L::packing, S> type;
};

/// Determine whether a block type is compatible with a requested layout.
/// The block's type information may lack information
/// that is necessary to determine compatibility at
/// compile-time. Therefore we provide a may-be-compatible
/// as a lower bound to a fully deterministic 
/// is-compatible metafunction.
///
/// Template parameters:
///
/// :B: the block type
/// :L: the requested layout
template <typename B, typename L>
struct maybe_layout_compatible
{
  typedef typename L::order_type req_order_type;
  static pack_type const req_packing = L::packing;
  static storage_format_type const req_storage_format = L::storage_format;

  typedef typename get_block_layout<B>::type block_layout_type;
  typedef typename block_layout_type::order_type act_order_type;
  static pack_type const act_packing = block_layout_type::packing;
  static storage_format_type const act_storage_format = block_layout_type::storage_format;

  static bool const value =
    (L::dim == B::dim ||
     (L::dim == 1 && act_packing == dense)) &&
    adjust_type<req_order_type, act_order_type>::compatible &&
    adjust_storage_format<req_storage_format, act_storage_format>::compatible;
};

template <typename B, typename L>
struct is_layout_compatible
{
  static bool const value = 
    maybe_layout_compatible<B, L>::value &&
    adjust_packing<L::packing, get_block_layout<B>::packing>::compatible;
};

} // namespace ovxx

#endif

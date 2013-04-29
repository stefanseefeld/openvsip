//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_OPT_ADJUST_LAYOUT_HPP
#define VSIP_OPT_ADJUST_LAYOUT_HPP

#include <vsip/core/layout.hpp>

namespace vsip
{
namespace impl
{

/// Adjust a type.  If a preferred type is given, use that.  Otherwise
/// if 'Any_type' is given, use the actual type.
///
/// Provides:
///  - type  - typedef with the adjusted type.
///  - equiv - bool set to true if the preferred and actual types
///            were equivalent (either the same, or preferred was
///            'Any_type').
template <typename PreferredT,
	  typename ActualT>
struct adjust_type
{
  typedef PreferredT type;
  static bool const equiv = false;
};

template <typename ActualT>
struct adjust_type<Any_type, ActualT>
{
  typedef ActualT type;
  static bool const equiv = true;
};

template <typename SameT>
struct adjust_type<SameT, SameT>
{
  typedef SameT type;
  static bool const equiv = true;
};

/// Adjust actual packing to preferred packing:
/// If preferred is 'unknown', keep actual packing unless it is
/// unknown (in which case it is set to 'dense').
/// Otherwise use preferred.
template <pack_type Preferred, pack_type Actual>
struct adjust_packing
{
  static pack_type const value = 
    Preferred == any_packing
    ? (Actual == any_packing ? dense : Actual)
    : Preferred;
  static bool const equiv =
    Preferred == Actual || Preferred == any_packing;
};

template <typename T,
	  storage_format_type Preferred,
	  storage_format_type Actual>
struct adjust_storage_format
{
  // For non-complex types, the complex-format is ignored.
  static storage_format_type const value = Actual;
  static bool const equiv = true;
};

template <typename T,
	  storage_format_type Preferred,
	  storage_format_type Actual>
struct adjust_storage_format<complex<T>, Preferred, Actual>
{
  static storage_format_type const value = Preferred;
  static bool const equiv = Preferred == Actual;
};

/// Adjust an actual layout policy against a required layout policy.
///
/// The resulting layout takes its values from the required layout
/// where it specifies a value, or from the actual layout when the
/// required layout specifies 'Any_type'.
template <typename T, typename Req, typename Act>
struct adjust_layout
{
  static dimension_type const dim = Req::dim;

  typedef typename Req::order_type req_order_type;
  static pack_type const req_packing = Req::packing;
  static storage_format_type const req_storage_format = Req::storage_format;

  typedef typename Act::order_type act_order_type;
  static pack_type const act_packing = Act::packing;
  static storage_format_type const act_storage_format = Act::storage_format;

  typedef typename adjust_type<req_order_type, act_order_type>::type order_type;
  static pack_type const packing = adjust_packing<req_packing, act_packing>::value;

  static storage_format_type const storage_format = 
    adjust_storage_format<T, req_storage_format, act_storage_format>::value;

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

template <storage_format_type C, typename L>
struct adjust_layout_storage_format
{
  typedef Layout<L::dim, typename L::order_type, L::packing, C> type;
};

/// Determine if an given layout policy is compatible with a required
/// layout policy.
///
/// The value_type 'T' is used to determine when differences between
/// storage_format matter.
template <typename T, typename Req, typename Act>
struct is_layout_compatible
{
  typedef typename Req::order_type req_order_type;
  static pack_type const req_packing = Req::packing;
  static storage_format_type const req_storage_format = Req::storage_format;

  typedef typename Act::order_type act_order_type;
  static pack_type const act_packing = Act::packing;
  static storage_format_type const act_storage_format = Act::storage_format;

  static bool const value =
    Req::dim == Act::dim &&
    adjust_type<req_order_type, act_order_type>::equiv &&
    adjust_packing<req_packing, act_packing>::equiv  &&
    adjust_storage_format<T, req_storage_format, act_storage_format>::equiv;
};

} // namespace vsip::impl
} // namespace vsip

#endif

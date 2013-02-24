/* Copyright (c) 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/adjust_layout.hpp
    @author  Jules Bergmann
    @date    2006-02-02
    @brief   VSIPL++ Library: Utilities to adjust layout policies.
*/

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
struct Adjust_type
{
  typedef PreferredT type;
  static bool const equiv = false;
};

template <typename ActualT>
struct Adjust_type<Any_type, ActualT>
{
  typedef ActualT type;
  static bool const equiv = true;
};

template <typename SameT>
struct Adjust_type<SameT, SameT>
{
  typedef SameT type;
  static bool const equiv = true;
};



/// Variant of Adjust_type for adjusting pack_type's.
///
/// In particular, adjusts Stride_unknown pack types to something that
/// can be used to allocate a block.
template <typename PreferredT,
	  typename ActualT>
struct Adjust_pack_type
  : Adjust_type<PreferredT, ActualT>
{};

template <>
struct Adjust_pack_type<Any_type, Stride_unknown>
{
  typedef Stride_unit_dense type;
  static bool const equiv = true;
};



/// Variant of Adjust_type for adjusting complex_type's.
///
/// Uses the value type T to determine when the complex_type matters.
/// In particular, for non-complex value types, the complex_type is
/// not applicable.
template <typename T,
	  typename PreferredT,
	  typename ActualT>
struct Adjust_complex_type
{
  typedef ActualT type;
  static bool const equiv = true;
};

template <typename T,
	  typename PreferredT,
	  typename ActualT>
struct Adjust_complex_type<complex<T>, PreferredT, ActualT>
  : Adjust_type<PreferredT, ActualT>
{};



/// Adjust an actual layout policy against a required layout policy.
///
/// The resulting layout takes its values from the required layout
/// where it specifies a value, or from the actual layout when the
/// required layout specifies 'Any_type'.
template <typename T,
	  typename RequiredLP,
	  typename ActualLP>
struct Adjust_layout
{
  static dimension_type const dim = RequiredLP::dim;

  typedef typename RequiredLP::order_type   req_order_t;
  typedef typename RequiredLP::pack_type    req_pack_t;
  typedef typename RequiredLP::complex_type req_complex_t;

  typedef typename ActualLP::order_type     act_order_t;
  typedef typename ActualLP::pack_type      act_pack_t;
  typedef typename ActualLP::complex_type   act_complex_t;

  typedef typename Adjust_type<req_order_t,     act_order_t>::type order_type;
  typedef typename Adjust_pack_type<req_pack_t, act_pack_t>::type  pack_type;

  typedef typename Adjust_complex_type<T, req_complex_t, act_complex_t>::type
		complex_type;

  typedef Layout<dim, order_type, pack_type, complex_type> type;
};



template <dimension_type NewDim,
	  typename       LP>
struct Adjust_layout_dim
{
  typedef typename LP::order_type     order_type;
  typedef typename LP::pack_type      pack_type;
  typedef typename LP::complex_type   complex_type;

  typedef Layout<NewDim, order_type, pack_type, complex_type> type;
};



template <typename NewPackType,
	  typename LP>
struct Adjust_layout_pack
{
  typedef typename LP::order_type     order_type;
  typedef typename LP::pack_type      pack_type;
  typedef typename LP::complex_type   complex_type;

  typedef Layout<LP::dim, order_type, NewPackType, complex_type> type;
};



template <typename NewComplexType,
	  typename LP>
struct Adjust_layout_complex
{
  typedef typename LP::order_type     order_type;
  typedef typename LP::pack_type      pack_type;

  typedef Layout<LP::dim, order_type, pack_type, NewComplexType> type;
};



/// Determine if an given layout policy is compatible with a required
/// layout policy.
///
/// The value_type 'T' is used to determine when differences between
/// complex_type matter.
template <typename T,
	  typename RequiredLP,
	  typename ActualLP>
struct Is_layout_compatible
{
  typedef typename RequiredLP::order_type   req_order_type;
  typedef typename RequiredLP::pack_type    req_pack_type;
  typedef typename RequiredLP::complex_type req_complex_type;

  typedef typename ActualLP::order_type     act_order_type;
  typedef typename ActualLP::pack_type      act_pack_type;
  typedef typename ActualLP::complex_type   act_complex_type;

  static bool const value =
    RequiredLP::dim == ActualLP::dim                     &&
    Adjust_type<           req_order_type,   act_order_type>::equiv &&
    Adjust_pack_type<      req_pack_type,    act_pack_type>::equiv  &&
    Adjust_complex_type<T, req_complex_type, act_complex_type>::equiv;
};

} // namespace vsip::impl
} // namespace vsip

#endif

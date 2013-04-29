/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   VSIPL++ Library: Template mechanism to choose the appropriate
///   access tag for a block, based on its layout policies and the
///   requested layout policies.
///
///   Used by vsip::dda::impl::Choose_access

#ifndef VSIP_IMPL_CHOOSE_ACCESS_HPP
#define VSIP_IMPL_CHOOSE_ACCESS_HPP

#include <vsip/core/metaprogramming.hpp>

namespace vsip
{
namespace dda
{
namespace impl
{
using namespace vsip::impl;

/// Specialized If-Then-Else class for Choose_asses.  Chooses either
/// IfType::type or ElseType::type based on boolean predicate.  Also
/// track reason_type.
template <bool     Predicate,
	  typename IfType,
	  typename ElseType>
struct ITE_type_reason;

template <typename IfType,
	  typename ElseType>
struct ITE_type_reason<true, IfType, ElseType>
{
  typedef typename IfType::type        type;
  typedef typename IfType::reason_type reason_type;
};

template <typename IfType,
	  typename ElseType>
struct ITE_type_reason<false, IfType, ElseType>
{
  typedef typename ElseType::type        type;
  typedef typename ElseType::reason_type reason_type;
};



/// Wrap a type so that it can be accessed via ::type.

template <typename T, typename ReasonT>
struct As_type_reason
{
  typedef T       type;
  typedef ReasonT reason_type;
};



/// Check if two Pack_types are
///  (a) both packing::aligned, and
///  (b) have the same alignment.

template <pack_type P1, pack_type P2>
struct CA_Equal_stride_unit_align
{
  static bool const value = P1 == P2 && P1 >= aligned_8 && P1 <= aligned_1024;
};

// Reason tags.

struct CA_Eq_cmplx_eq_order_unknown_stride_ok;
struct CA_Eq_cmplx_eq_order_unit_stride_ok;
struct CA_Eq_cmplx_eq_order_dense_ok;
struct CA_Eq_cmplx_eq_order_aligned_ok;
struct CA_Eq_cmplx_eq_order_different_stride;
struct CA_Eq_cmplx_different_dim_order_but_both_dense;
struct CA_Eq_cmplx_different_dim_order;
struct CA_General_different_complex_layout;



template <typename Demotion, pack_type Pack1, pack_type Pack2>
struct CA_Eq_cmplx_eq_order
{
  typedef CA_Eq_cmplx_eq_order_unknown_stride_ok    reason1_type;
  typedef CA_Eq_cmplx_eq_order_unit_stride_ok       reason2_type;
  typedef CA_Eq_cmplx_eq_order_dense_ok reason3_type;
  typedef CA_Eq_cmplx_eq_order_aligned_ok reason4_type;
  typedef CA_Eq_cmplx_eq_order_different_stride     reason5_type;

  typedef typename Demotion::direct_type  direct_type;
  typedef typename Demotion::reorder_type reorder_type;
  typedef typename Demotion::flex_type    flex_type;

  typedef 
  ITE_type_reason<Pack2 == any_packing,
		  As_type_reason<direct_type, reason1_type>,

	ITE_type_reason<Pack2 == unit_stride && Pack1 != any_packing,
			As_type_reason<direct_type, reason2_type>,

	ITE_type_reason<Pack2 == dense && Pack1 == dense,
			As_type_reason<direct_type, reason3_type>,

  ITE_type_reason<CA_Equal_stride_unit_align<Pack1, Pack2>::value,
	As_type_reason<direct_type, reason4_type>,

	As_type_reason<flex_type, reason5_type> > > > > ite_type;

  typedef typename ite_type::type        type;
  typedef typename ite_type::reason_type reason_type;
};


template <typename       Demotion,
	  typename       Order1,
	  pack_type Pack1,
	  typename       Order2,
	  pack_type Pack2>
struct CA_Eq_cmplx
{
  typedef CA_Eq_cmplx_different_dim_order_but_both_dense reason1_type;
  typedef CA_Eq_cmplx_different_dim_order                reason2_type;

  typedef typename Demotion::reorder_type reorder_type;
  typedef typename Demotion::flex_type    flex_type;

  // If layouts have different dimension-order, then
  //   reorder access if they are both dense,
  //   copy access    otherwise
  typedef 
          ITE_type_reason<is_same<Order1, Order2>::value,
		CA_Eq_cmplx_eq_order<Demotion, Pack1, Pack2>,
		ITE_type_reason<Pack1 == dense && Pack2 == dense,
                        As_type_reason<reorder_type, reason1_type>,
			As_type_reason<flex_type, reason2_type>
                        >
                > ite_type;

  typedef typename ite_type::type        type;
  typedef typename ite_type::reason_type reason_type;
};



template <typename       Demotion,
	  typename       Order1,
	  pack_type Pack1,
	  storage_format_type Cmplx1,
	  typename       Order2,
	  pack_type Pack2,
	  storage_format_type Cmplx2>
struct CA_General
{
  typedef CA_General_different_complex_layout my_reason_type;

  // If layouts do not have same complex layout, then copy access
  typedef
          ITE_type_reason<Cmplx1 == Cmplx2,
			  CA_Eq_cmplx<Demotion, Order1, Pack1, Order2, Pack2>,
			  As_type_reason<typename Demotion::copy_type,
					 my_reason_type> >
		ite_type;

  typedef typename ite_type::type        type;
  typedef typename ite_type::reason_type reason_type;
};

} // namespace vsip::dda::impl
} // namespace vsip::dda
} // namespace vsip

#endif

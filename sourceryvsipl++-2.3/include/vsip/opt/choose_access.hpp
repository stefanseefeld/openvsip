/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/choose_access.hpp
    @author  Jules Bergmann
    @date    2005-04-13
    @brief   VSIPL++ Library: Template mechanism to choose the appropriate
	     access tag for a block, based on its layout policies and the
	     requested layout policies.

	     Used by vsip::impl::Choose_access

*/

#ifndef VSIP_IMPL_CHOOSE_ACCESS_HPP
#define VSIP_IMPL_CHOOSE_ACCESS_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/metaprogramming.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace choose_access
{



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
///  (a) both Stride_unit_align, and
///  (b) have the same alignment.

template <typename PackType1,
	  typename PackType2>
struct CA_Equal_stride_unit_align
{
  static bool const value = false;
};

template <unsigned Align>
struct CA_Equal_stride_unit_align<Stride_unit_align<Align>, Stride_unit_align<Align> >
{
  static bool const value = true;
};



// Reason tags.

struct CA_Eq_cmplx_eq_order_unknown_stride_ok;
struct CA_Eq_cmplx_eq_order_unit_stride_ok;
struct CA_Eq_cmplx_eq_order_unit_stride_dense_ok;
struct CA_Eq_cmplx_eq_order_unit_stride_align_ok;
struct CA_Eq_cmplx_eq_order_different_stride;
struct CA_Eq_cmplx_different_dim_order_but_both_dense;
struct CA_Eq_cmplx_different_dim_order;
struct CA_General_different_complex_layout;



template <typename       Demotion,
	  typename       Pack1,
	  typename       Pack2>
struct CA_Eq_cmplx_eq_order
{
  typedef CA_Eq_cmplx_eq_order_unknown_stride_ok    reason1_type;
  typedef CA_Eq_cmplx_eq_order_unit_stride_ok       reason2_type;
  typedef CA_Eq_cmplx_eq_order_unit_stride_dense_ok reason3_type;
  typedef CA_Eq_cmplx_eq_order_unit_stride_align_ok reason4_type;
  typedef CA_Eq_cmplx_eq_order_different_stride     reason5_type;

  typedef typename Demotion::direct_type  direct_type;
  typedef typename Demotion::reorder_type reorder_type;
  typedef typename Demotion::flex_type    flex_type;

  typedef 
  ITE_type_reason<Type_equal<Pack2, Stride_unknown>::value,
	As_type_reason<direct_type, reason1_type>,

  ITE_type_reason< Type_equal<Pack2, Stride_unit>::value &&
                  !Type_equal<Pack1, Stride_unknown>::value,
	As_type_reason<direct_type, reason2_type>,

  ITE_type_reason<Type_equal<Pack2, Stride_unit_dense>::value &&
                  Type_equal<Pack1, Stride_unit_dense>::value,
	As_type_reason<direct_type, reason3_type>,

  ITE_type_reason<CA_Equal_stride_unit_align<Pack1, Pack2>::value,
	As_type_reason<direct_type, reason4_type>,

	As_type_reason<flex_type, reason5_type> > > > > ite_type;

  typedef typename ite_type::type        type;
  typedef typename ite_type::reason_type reason_type;
};


template <typename       Demotion,
	  typename       Order1,
	  typename       Pack1,
	  typename       Order2,
	  typename       Pack2>
struct CA_Eq_cmplx
{
  typedef CA_Eq_cmplx_different_dim_order_but_both_dense reason1_type;
  typedef CA_Eq_cmplx_different_dim_order                reason2_type;

  typedef typename Demotion::reorder_type reorder_type;
  typedef typename Demotion::flex_type    flex_type;

  // If layouts have different dimension-ordering, then
  //   reorder access if they are both dense,
  //   copy access    otherwise
  typedef 
          ITE_type_reason<Type_equal<Order1, Order2>::value,
		CA_Eq_cmplx_eq_order<Demotion, Pack1, Pack2>,
		ITE_type_reason<Type_equal<Pack1, Stride_unit_dense>::value &&
                                Type_equal<Pack2, Stride_unit_dense>::value,
                        As_type_reason<reorder_type, reason1_type>,
			As_type_reason<flex_type, reason2_type>
                        >
                > ite_type;

  typedef typename ite_type::type        type;
  typedef typename ite_type::reason_type reason_type;
};



template <typename       Demotion,
	  typename       Order1,
	  typename       Pack1,
	  typename       Cmplx1,
	  typename       Order2,
	  typename       Pack2,
	  typename       Cmplx2>
struct CA_General
{
  typedef CA_General_different_complex_layout my_reason_type;

  // If layouts do not have same complex layout, then copy access
  typedef
          ITE_type_reason<Type_equal<Cmplx1, Cmplx2>::value,
			  CA_Eq_cmplx<Demotion, Order1, Pack1, Order2, Pack2>,
			  As_type_reason<typename Demotion::copy_type,
					 my_reason_type> >
		ite_type;

  typedef typename ite_type::type        type;
  typedef typename ite_type::reason_type reason_type;
};

} // namespace vsip::impl::choose_access



} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_CHOOSE_ACCESS_HPP

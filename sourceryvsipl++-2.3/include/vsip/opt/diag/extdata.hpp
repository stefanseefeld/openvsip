/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/diag/eval_vcmp.hpp
    @author  Jules Bergmann
    @date    2007-03-06
    @brief   VSIPL++ Library: Diagnostics for extdata.
*/

#ifndef VSIP_OPT_DIAG_EXTDATA_HPP
#define VSIP_OPT_DIAG_EXTDATA_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <iomanip>

#include <vsip/opt/diag/class_name.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{



namespace diag_detail
{

VSIP_IMPL_CLASS_NAME(Direct_access_tag)
VSIP_IMPL_CLASS_NAME(Reorder_access_tag)
VSIP_IMPL_CLASS_NAME(Copy_access_tag)
VSIP_IMPL_CLASS_NAME(Flexible_access_tag)
VSIP_IMPL_CLASS_NAME(Bogus_access_tag)
VSIP_IMPL_CLASS_NAME(Default_access_tag)

VSIP_IMPL_CLASS_NAME(choose_access::CA_Eq_cmplx_eq_order_unknown_stride_ok)
VSIP_IMPL_CLASS_NAME(choose_access::CA_Eq_cmplx_eq_order_unit_stride_ok)
VSIP_IMPL_CLASS_NAME(choose_access::CA_Eq_cmplx_eq_order_unit_stride_dense_ok)
VSIP_IMPL_CLASS_NAME(choose_access::CA_Eq_cmplx_eq_order_unit_stride_align_ok)
VSIP_IMPL_CLASS_NAME(choose_access::CA_Eq_cmplx_eq_order_different_stride)
VSIP_IMPL_CLASS_NAME(choose_access::CA_Eq_cmplx_different_dim_order_but_both_dense)
VSIP_IMPL_CLASS_NAME(choose_access::CA_Eq_cmplx_different_dim_order)
VSIP_IMPL_CLASS_NAME(choose_access::CA_General_different_complex_layout)

VSIP_IMPL_CLASS_NAME(Cmplx_inter_fmt)
VSIP_IMPL_CLASS_NAME(Cmplx_split_fmt)

} // namespace vsip::impl::diag_detail



template <typename BlockT,
	  typename LP  = typename Desired_block_layout<BlockT>::layout_type>
struct Diagnose_ext_data
{
  typedef Choose_access<BlockT, LP> ca_type;
  typedef typename ca_type::type        access_type;
  typedef typename ca_type::reason_type reason_type;

  typedef typename Block_layout<BlockT>::complex_type blk_complex_type;
  typedef typename LP::complex_type lp_complex_type;

  static void diag(std::string name)
  {
    using diag_detail::Class_name;
    using std::cout;
    using std::endl;

    cout << "diagnose_ext_data(" << name << ")" << endl
	 << "  BlockT: " << typeid(BlockT).name() << endl
	 << "  Block LP" << endl
	 << "    complex_type: " << Class_name<blk_complex_type>::name() << endl
	 << "  Req LP" << endl
	 << "    complex_type: " << Class_name<lp_complex_type>::name() << endl
	 << "  access_type: " << Class_name<access_type>::name() << endl
	 << "  reason_type: " << Class_name<reason_type>::name() << endl
	 << "  static-cost: " << data_access::Cost<access_type>::value << endl
      ;
  }
};



template <typename BlockT,
	  dimension_type Dim = Block_layout<BlockT>::dim>
struct Diagnose_rt_ext_data
{
  typedef typename Block_layout<BlockT>::access_type              AT;
  typedef data_access::Rt_low_level_data_access<AT, BlockT, Dim>   ext_type;

  static void diag(std::string name)
  {
    using diag_detail::Class_name;
    using std::cout;
    using std::endl;

    cout << "diagnose_rt_ext_data(" << name << ")" << endl
	 << "  BlockT: " << typeid(BlockT).name() << endl
	 << "  access_type: " << Class_name<AT>::name() << endl
      // << "  static-cost: " << access_type::cost << endl
      ;
  }
};

} // namespace vsip::impl::diag_detail
} // namespace vsip

#endif // VSIP_OPT_DIAG_EXTDATA_HPP

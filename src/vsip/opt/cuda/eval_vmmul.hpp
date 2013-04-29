/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/cuda/eval_vmmul.hpp
    @author  Don McCoy
    @date    2009-04-07
    @brief   VSIPL++ Library: CUDA evaluator for vector-matrix multiply.

*/

#ifndef VSIP_OPT_CUDA_EVAL_VMMUL_HPP
#define VSIP_OPT_CUDA_EVAL_VMMUL_HPP

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cuda/library.hpp>
#include <vsip/opt/cuda/vmmul.hpp>
#include <vsip/opt/cuda/dda.hpp>

namespace vsip_csl
{
namespace dispatcher
{

/// Evaluator for vector-matrix multiply.
///
/// Dispatches cases where the dimension order matches the 
/// requested orientation to the SPU's (row-major/by-row and 
/// col-major/by-col).  The other cases are re-dispatched.
template <typename LHS,
	  typename VBlock,
	  typename MBlock,
	  dimension_type SD>
struct Evaluator<op::assign<2>, be::cuda,
		 void(LHS &, expr::Vmmul<SD, VBlock, MBlock> const &)>
{
  static char const* name() { return "CUDA_vmmul"; }

  typedef expr::Vmmul<SD, VBlock, MBlock> RHS;
  typedef typename RHS::value_type rhs_type;
  typedef typename LHS::value_type lhs_type;
  typedef typename VBlock::value_type v_type;
  typedef typename MBlock::value_type m_type;
  typedef typename get_block_layout<LHS>::type lhs_lp;
  typedef typename get_block_layout<VBlock>::type vblock_lp;
  typedef typename get_block_layout<MBlock>::type mblock_lp;
  typedef typename get_block_layout<RHS>::order_type order_type;
  typedef typename get_block_layout<MBlock>::order_type src_order_type;

  static bool const ct_valid = 
    // inputs must not be expression blocks
    impl::is_expr_block<VBlock>::value == 0 &&
    impl::is_expr_block<MBlock>::value == 0 &&
    // split complex not supported
    impl::is_split_block<LHS>::value == 0 &&
    impl::is_split_block<VBlock>::value == 0 &&
    impl::is_split_block<MBlock>::value == 0 &&
    // ensure value types are supported
    impl::cuda::Traits<lhs_type>::valid &&
    impl::cuda::Traits<v_type>::valid &&
    impl::cuda::Traits<m_type>::valid &&
    // result type must match type expected (determined by promotion)
    is_same<lhs_type, rhs_type>::value &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<VBlock, dda::in>::ct_cost == 0 &&
    dda::Data<MBlock, dda::in>::ct_cost == 0 &&
    // dimension order must be the same
    is_same<order_type, src_order_type>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    VBlock const &vblock = rhs.get_vblk();
    MBlock const &mblock = rhs.get_mblk();

    dda::Data<LHS, dda::out, lhs_lp>  data_lhs(lhs);
    dda::Data<VBlock, dda::in, vblock_lp> data_v(vblock);
    dda::Data<MBlock, dda::in, mblock_lp> data_m(mblock);

    if ((SD == row && is_same<order_type, row2_type>::value) ||
        (SD == col && is_same<order_type, col2_type>::value))
    {
      dimension_type const axis = SD == row ? 1 : 0;
      length_type lhs_stride = static_cast<length_type>(abs(data_lhs.stride(axis == 0)));
      length_type m_stride = static_cast<length_type>(abs(data_m.stride(axis == 0)));
      return 
        // make sure blocks are dense (major stride == minor size)
        (data_lhs.size(axis) == lhs_stride) &&
        (data_m.size(axis) == m_stride) &&
        // ensure unit stride along the dimension opposite the chosen one
	(data_lhs.stride(axis) == 1) &&
	(data_m.stride(axis) == 1) &&
	(data_v.stride(0) == 1);
    }
    else
    {
      dimension_type const axis = SD == row ? 0 : 1;
      length_type lhs_stride = static_cast<length_type>(abs(data_lhs.stride(axis == 0)));
      length_type m_stride = static_cast<length_type>(abs(data_m.stride(axis == 0)));
      return 
        // make sure blocks are dense (major stride == minor size)
        (data_lhs.size(axis) == lhs_stride) &&
        (data_m.size(axis) == m_stride) &&
        // ensure unit stride along the same dimension as the chosen one
	(data_lhs.stride(axis) == 1) &&
	(data_m.stride(axis) == 1) &&
	(data_v.stride(0) == 1);
    }
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    VBlock const &vblock = rhs.get_vblk();
    MBlock const &mblock = rhs.get_mblk();

    impl::cuda::dda::Data<LHS, dda::out> dev_lhs(lhs);
    impl::cuda::dda::Data<VBlock, dda::in> dev_v(vblock);
    impl::cuda::dda::Data<MBlock, dda::in> dev_m(mblock);

    // The ct_valid check above ensures that the order taken 
    // matches the storage order if reaches this point.
    if (SD == row && is_same<order_type, row2_type>::value)
    {
      impl::cuda::vmmul_row(dev_v.ptr(),
			    dev_m.ptr(),
			    dev_lhs.ptr(),
			    lhs.size(2, 0),    // number of rows
			    lhs.size(2, 1));   // length of each row
    }
    else if (SD == col && is_same<order_type, row2_type>::value)
    {
      impl::cuda::vmmul_col(dev_v.ptr(),
			    dev_m.ptr(),
			    dev_lhs.ptr(),
			    lhs.size(2, 0),    // number of rows
			    lhs.size(2, 1));   // length of each row
    }
    else if (SD == col && is_same<order_type, col2_type>::value)
    {
      impl::cuda::vmmul_row(dev_v.ptr(),
			    dev_m.ptr(),
			    dev_lhs.ptr(),
			    lhs.size(2, 1),    // number of cols
			    lhs.size(2, 0));   // length of each col
    }
    else // if (SD == row && is_same<order_type, col2_type>::value)
    {
      impl::cuda::vmmul_col(dev_v.ptr(),
			    dev_m.ptr(),
			    dev_lhs.ptr(),
			    lhs.size(2, 1),    // number of cols
			    lhs.size(2, 0));   // length of each col
    }
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_CUDA_EVAL_VMMUL_HPP

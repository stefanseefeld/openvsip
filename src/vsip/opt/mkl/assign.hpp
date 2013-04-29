/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef vsip_opt_mkl_assign_hpp_
#define vsip_opt_mkl_assign_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/mkl/unary.hpp>
#include <vsip/opt/mkl/binary.hpp>
#include <vsip/dda.hpp>

namespace vsip_csl
{
namespace dispatcher
{
/// Specialization for unary elementwise operations
template <typename LHS, template <typename> class Operator, typename Block>
struct Evaluator<op::assign<1>, be::intel_ipp,
		 void(LHS &, expr::Unary<Operator, Block, true> const &)>
{
  typedef expr::Unary<Operator, Block, true> RHS;

  typedef typename impl::adjust_layout_dim<
  1, typename get_block_layout<LHS>::type>::type
    lhs_layout;
  typedef dda::Data<LHS, dda::out, lhs_layout> lhs_dda_type;

  typedef typename impl::adjust_layout_dim<
  1, typename get_block_layout<Block>::type>::type
    block_layout;
  typedef dda::Data<Block, dda::in, block_layout> block_dda_type;

  typedef impl::mkl::Unary<Operator,
    void(typename block_dda_type::ptr_type,
	 typename lhs_dda_type::ptr_type,
	 length_type)> operation_type;

  static char const *name() { return operation_type::name();}

  static bool const ct_valid = 
    !impl::is_expr_block<Block>::value &&
    operation_type::is_supported &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<Block, dda::in>::ct_cost == 0 &&
     /* MKL does not support complex split */
     !impl::is_split_block<LHS>::value &&
     !impl::is_split_block<Block>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    typedef Operator<typename Block::value_type> op_type;
    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_layout>  data_lhs(lhs);
    dda::Data<Block, dda::in, block_layout> data_b(rhs.arg());
    return data_lhs.stride(0) == 1 && data_b.stride(0) == 1;
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    lhs_dda_type lhs_dda(lhs);
    block_dda_type block_dda(rhs.arg());
    operation_type::exec(block_dda.ptr(), lhs_dda.ptr(), lhs.size());
  }
};

/// Specialization for binary elementwise operations
template <typename LHS,
          template <typename, typename> class Operator,
	  typename LBlock,
	  typename RBlock>
struct Evaluator<op::assign<1>, be::intel_ipp,
		 void(LHS &, expr::Binary<Operator, LBlock, RBlock, true> const &)>
{
  typedef expr::Binary<Operator, LBlock, RBlock, true> RHS;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<LHS>::type>::type
    lhs_layout;
  typedef dda::Data<LHS, dda::out, lhs_layout> lhs_dda_type;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<LBlock>::type>::type
    lblock_layout;
  typedef dda::Data<LBlock, dda::in, lblock_layout> lblock_dda_type;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<RBlock>::type>::type
    rblock_layout;
  typedef dda::Data<RBlock, dda::in, rblock_layout> rblock_dda_type;

  typedef impl::mkl::Binary<Operator,
    void(typename lblock_dda_type::ptr_type,
	 typename rblock_dda_type::ptr_type,
	 typename lhs_dda_type::ptr_type,
	 length_type)> operation_type;

  static char const *name() { return operation_type::name();}

  static bool const ct_valid = 
    !impl::is_expr_block<LBlock>::value &&
    !impl::is_expr_block<RBlock>::value &&
    operation_type::is_supported &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<LBlock, dda::in>::ct_cost == 0 &&
    dda::Data<RBlock, dda::in>::ct_cost == 0;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_layout>  data_lhs(lhs);
    dda::Data<LBlock, dda::in, lblock_layout> data_l(rhs.arg1());
    dda::Data<RBlock, dda::in, rblock_layout> data_r(rhs.arg2());
    return (data_lhs.stride(0) == 1 &&
	    data_l.stride(0) == 1 &&
	    data_r.stride(0) == 1);
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    lhs_dda_type lhs_dda(lhs);
    lblock_dda_type lblock_dda(rhs.arg1());
    rblock_dda_type rblock_dda(rhs.arg2());
    operation_type::exec(lblock_dda.ptr(),
			 rblock_dda.ptr(),
			 lhs_dda.ptr(), lhs.size());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

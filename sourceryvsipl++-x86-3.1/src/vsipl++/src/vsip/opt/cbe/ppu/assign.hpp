/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_OPT_CBE_PPU_ASSIGN_HPP
#define VSIP_OPT_CBE_PPU_ASSIGN_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/cbe/cml/transpose.hpp>
#include <vsip/opt/cbe/ppu/bindings.hpp>
#include <vsip/opt/cbe/ppu/unary.hpp>
#include <vsip/opt/cbe/ppu/binary.hpp>
#include <vsip/opt/cbe/ppu/ternary.hpp>
#include <vsip/opt/cbe/ppu/logma.hpp>
#include <vsip/opt/cbe/ppu/eval_fastconv.hpp>

namespace vsip
{
namespace impl
{
namespace cbe
{
template <template <typename> class Operator,
	  typename LHS,
	  typename Block>
struct Unary_evaluator
{
  typedef expr::Unary<Operator, Block, true> RHS;

  typedef typename adjust_layout_dim<
      1, typename get_block_layout<LHS>::type>::type
    lhs_lp;

  typedef typename adjust_layout_dim<
      1, typename get_block_layout<Block>::type>::type
    block_lp;

  static bool const ct_valid = 
    !is_expr_block<Block>::value &&
    Is_un_op_supported<Operator,
		       typename LHS::value_type,
		       impl::is_split_block<LHS>::value,
		       typename Block::value_type,
		       impl::is_split_block<Block>::value>::value &&
     // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<Block, dda::in>::ct_cost == 0;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    typedef Operator<typename Block::value_type> operation_type;
    // Note that we assume the argument list doesn't mix interleaved and
    // split complex.
    typedef Size_threshold<operation_type,
                           impl::is_split_block<LHS>::value> threshold_type;

    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_lp>  data_lhs(lhs);
    dda::Data<Block, dda::in, block_lp> data_b(rhs.arg());
    return data_lhs.size(0) >= threshold_type::value &&
           data_lhs.stride(0) == 1 &&
	   data_b.stride(0) == 1   &&
	   is_dma_addr_ok(data_lhs.ptr()) &&
	   is_dma_addr_ok(data_b.ptr())   &&
           Task_manager::instance()->num_spes() > 0;
  }
};

template <template <typename, typename> class Operator,
	  typename LHS,
	  typename LBlock,
	  typename RBlock>
struct Binary_evaluator
{
  typedef expr::Binary<Operator, LBlock, RBlock, true> RHS;

  typedef typename adjust_layout_dim<
      1, typename get_block_layout<LHS>::type>::type
    lhs_lp;

  typedef typename adjust_layout_dim<
      1, typename get_block_layout<LBlock>::type>::type
    lblock_lp;

  typedef typename adjust_layout_dim<
      1, typename get_block_layout<RBlock>::type>::type
    rblock_lp;

  static bool const ct_valid = 
    !is_expr_block<LBlock>::value &&
    !is_expr_block<RBlock>::value &&
    Is_bin_op_supported<Operator,
			typename LHS::value_type,
			impl::is_split_block<LHS>::value,
			typename LBlock::value_type,
			impl::is_split_block<LBlock>::value,
			typename RBlock::value_type,
			impl::is_split_block<RBlock>::value>::value &&
     // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<LBlock, dda::in>::ct_cost == 0 &&
    dda::Data<RBlock, dda::in>::ct_cost == 0;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    typedef Operator<typename LBlock::value_type,
                     typename RBlock::value_type> operation_type;
    // Note that we assume the argument list doesn't mix interleaved and
    // split complex.
    typedef Size_threshold<operation_type,
                           impl::is_split_block<LHS>::value> threshold_type;

    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_lp>  data_lhs(lhs);
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg1());
    dda::Data<RBlock, dda::in, rblock_lp> data_r(rhs.arg2());
    return data_lhs.size(0) >= threshold_type::value &&
           data_lhs.stride(0) == 1 &&
	   data_l.stride(0) == 1   &&
	   data_r.stride(0) == 1   &&
	   is_dma_addr_ok(data_lhs.ptr()) &&
	   is_dma_addr_ok(data_l.ptr())   &&
	   is_dma_addr_ok(data_r.ptr())   &&
           Task_manager::instance()->num_spes() > 0;
  }
};

template <template <typename, typename, typename> class Operator,
	  typename LHS,
	  typename Block1,
	  typename Block2,
	  typename Block3>
struct Ternary_evaluator
{
  typedef expr::Ternary<Operator, Block1, Block2, Block3, true> RHS;

  typedef typename adjust_layout_dim<
      1, typename get_block_layout<LHS>::type>::type
    lhs_lp;

  typedef typename adjust_layout_dim<
      1, typename get_block_layout<Block1>::type>::type
    block1_lp;

  typedef typename adjust_layout_dim<
      1, typename get_block_layout<Block2>::type>::type
    block2_lp;

  typedef typename adjust_layout_dim<
      1, typename get_block_layout<Block3>::type>::type
    block3_lp;

  static bool const ct_valid = 
    !is_expr_block<Block1>::value &&
    !is_expr_block<Block2>::value &&
    !is_expr_block<Block3>::value &&
    Is_tern_op_supported<Operator,
			 typename LHS::value_type,
			 impl::is_split_block<LHS>::value,
			 typename Block1::value_type,
			 impl::is_split_block<Block1>::value,
			 typename Block2::value_type,
			 impl::is_split_block<Block2>::value,
			 typename Block3::value_type,
			 impl::is_split_block<Block3>::value>::value &&
     // check that direct access is supported
     dda::Data<LHS, dda::out>::ct_cost == 0 &&
     dda::Data<Block1, dda::in>::ct_cost == 0 &&
     dda::Data<Block2, dda::in>::ct_cost == 0 &&
     dda::Data<Block3, dda::in>::ct_cost == 0;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    typedef Operator<typename Block1::value_type,
                     typename Block2::value_type,
                     typename Block3::value_type> operation_type;
    // Note that we assume the argument list doesn't mix interleaved and
    // split complex.
    typedef Size_threshold<operation_type,
                           impl::is_split_block<LHS>::value> threshold_type;

    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_lp>  data_lhs(lhs);
    dda::Data<Block1, dda::in, block1_lp> data_1(rhs.arg1());
    dda::Data<Block2, dda::in, block2_lp> data_2(rhs.arg2());
    dda::Data<Block3, dda::in, block3_lp> data_3(rhs.arg3());
    return data_lhs.size(0) >= threshold_type::value &&
           data_lhs.stride(0) == 1 &&
	   data_1.stride(0) == 1   &&
	   data_2.stride(0) == 1   &&
	   data_2.stride(0) == 1   &&
	   is_dma_addr_ok(data_lhs.ptr()) &&
	   is_dma_addr_ok(data_1.ptr())   &&
	   is_dma_addr_ok(data_2.ptr())   &&
	   is_dma_addr_ok(data_3.ptr())   &&
           Task_manager::instance()->num_spes() > 0;
  }
};

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip


namespace vsip_csl
{
namespace dispatcher
{

/// Unary expression evaluation.
#define V_EXPR(OP, FUN)              			                \
template <typename LHS, typename Block>		                        \
struct Evaluator<op::assign<1>, be::cbe_sdk,				\
		 void(LHS &,				                \
		      expr::Unary<OP, Block, true> const &)>	        \
  : impl::cbe::Unary_evaluator<OP, LHS, Block>                          \
{									\
  static char const* name() { return "Expr_CBE_SDK_V-" #FUN; }		\
									\
  typedef expr::Unary<OP, Block, true> RHS;		                \
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LHS>::type>::type			\
    lhs_lp;								\
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<Block>::type>::type		\
    block_lp;								\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);			\
    dda::Data<Block, dda::in, block_lp> data_b(rhs.arg());		\
									\
    FUN(data_b.ptr(), data_lhs.ptr(), lhs.size());	                \
  } 									\
};

V_EXPR(expr::op::Sqrt,   impl::cbe::vsqrt)
V_EXPR(expr::op::Minus,  impl::cbe::vminus)
V_EXPR(expr::op::Sq,     impl::cbe::vsq)
V_EXPR(expr::op::Mag,    impl::cbe::vmag)
V_EXPR(expr::op::Magsq,  impl::cbe::vmagsq)
V_EXPR(expr::op::Conj,   impl::cbe::vconj)
V_EXPR(expr::op::Atan,   impl::cbe::vatan)
V_EXPR(expr::op::Cos,    impl::cbe::vcos)
V_EXPR(expr::op::Sin,    impl::cbe::vsin)
V_EXPR(expr::op::Log,    impl::cbe::vlog)
V_EXPR(expr::op::Log10,  impl::cbe::vlog10)

#undef V_EXPR

/// Binary expression evaluation.
#define VV_EXPR(OP, FUN)              			                \
template <typename LHS, typename LBlock, typename RBlock>		\
struct Evaluator<op::assign<1>, be::cbe_sdk,				\
		 void(LHS &,				                \
		      expr::Binary<OP, LBlock, RBlock, true> const &)>	\
  : impl::cbe::Binary_evaluator<OP, LHS, LBlock, RBlock>		\
{									\
  static char const* name() { return "Expr_CBE_SDK_VV-" #FUN; }		\
									\
  typedef expr::Binary<OP, LBlock, RBlock, true> RHS;		        \
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LHS>::type>::type			\
    lhs_lp;								\
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LBlock>::type>::type		\
    lblock_lp;								\
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<RBlock>::type>::type		\
    rblock_lp;								\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);			\
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg1());		\
    dda::Data<RBlock, dda::in, rblock_lp> data_r(rhs.arg2());		\
									\
    FUN(data_l.ptr(), data_r.ptr(), data_lhs.ptr(), lhs.size());		\
  } 									\
};

VV_EXPR(expr::op::Add,   impl::cbe::vadd)
VV_EXPR(expr::op::Sub,   impl::cbe::vsub)
VV_EXPR(expr::op::Mult,  impl::cbe::vmul)
VV_EXPR(expr::op::Div,   impl::cbe::vdiv)
VV_EXPR(expr::op::Atan2, impl::cbe::vatan2)
VV_EXPR(expr::op::Hypot, impl::cbe::vhypot)

#undef VV_EXPR

/// Ternary expression evaluation.
#define VVV_EXPR(OP, FUN)              			                \
template <typename LHS,                                                 \
	  typename Block1, typename Block2, typename Block3>		\
struct Evaluator<op::assign<1>, be::cbe_sdk,				\
  void(LHS &,				                                \
       expr::Ternary<OP, Block1, Block2, Block3, true> const &)>	\
  : impl::cbe::Ternary_evaluator<OP, LHS, Block1, Block2, Block3>	\
{									\
  static char const* name() { return "Expr_CBE_SDK_VVV-" #FUN; }	\
									\
  typedef expr::Ternary<OP, Block1, Block2, Block3, true> RHS;		\
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LHS>::type>::type			\
    lhs_lp;								\
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<Block1>::type>::type		\
    block1_lp;								\
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<Block2>::type>::type		\
    block2_lp;								\
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<Block3>::type>::type		\
    block3_lp;								\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);			\
    dda::Data<Block1, dda::in, block1_lp> data_1(rhs.arg1());		\
    dda::Data<Block2, dda::in, block2_lp> data_2(rhs.arg2());		\
    dda::Data<Block3, dda::in, block3_lp> data_3(rhs.arg3());		\
									\
    FUN(data_1.ptr(), data_2.ptr(), data_3.ptr(), data_lhs.ptr(),	\
	lhs.size());							\
  } 									\
};

VVV_EXPR(expr::op::Am,   impl::cbe::vam)
VVV_EXPR(expr::op::Ma,   impl::cbe::vma)

#undef VVV_EXPR



/// Evaluator for log10(A) * b + c, log of a vector with scalar
/// multiply and accumulate.
template <typename LHS,
	  typename Block1, typename Block2, typename Block3>
struct Evaluator<op::assign<1>, be::cbe_sdk,
  void(LHS &,
    expr::Ternary<expr::op::Ma, 
      expr::Unary<expr::op::Log10, Block1, true> const,
      Block2, Block3, true> const &)>
{
  static char const* name() { return "Expr_CBE_SDK_VSS-vlma"; }

  typedef expr::Unary<expr::op::Log10, Block1, true> log10_type;

  typedef expr::Ternary<expr::op::Ma, 
    log10_type const, Block2, Block3, true> RHS;

  typedef typename vsip::impl::adjust_layout_dim<
    1, typename get_block_layout<LHS>::type>::type lhs_lp;

  typedef typename vsip::impl::adjust_layout_dim<
    1, typename get_block_layout<Block1>::type>::type block1_lp;
  

  static bool const ct_valid = 
    !vsip::impl::is_expr_block<Block1>::value &&
    vsip::impl::is_scalar_block<Block2>::value &&
    vsip::impl::is_scalar_block<Block3>::value &&
    // check that direct access is supported
    vsip::dda::Data<LHS, dda::out>::ct_cost == 0 &&
    vsip::dda::Data<Block1, dda::in>::ct_cost == 0;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    typedef vsip_csl::expr::op::Lma<
      typename Block1::value_type, 
      typename Block2::value_type, 
      typename Block3::value_type> operation_type;

    // Note that we assume the argument list doesn't mix interleaved and
    // split complex.  The check in ct_valid above ensures this.
    typedef vsip::impl::cbe::Size_threshold<operation_type,
      vsip::impl::is_split_block<LHS>::value> threshold_type;

    // check if all data is unit stride
    vsip::dda::Data<LHS, vsip::dda::out, lhs_lp>  data_lhs(lhs);
    vsip::dda::Data<Block1, vsip::dda::in, block1_lp> data_1(rhs.arg1().arg());
    return 
      data_lhs.size(0) >= threshold_type::value &&
      data_lhs.stride(0) == 1 &&
      data_1.stride(0) == 1   &&
      impl::cbe::is_dma_addr_ok(data_lhs.ptr()) &&
      impl::cbe::is_dma_addr_ok(data_1.ptr())   &&
      vsip::impl::cbe::Task_manager::instance()->num_spes() > 0;
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    vsip::dda::Data<LHS, vsip::dda::out, lhs_lp> data_lhs(lhs);
    vsip::dda::Data<Block1, vsip::dda::in, block1_lp> data_1(rhs.arg1().arg());

    vsip::impl::cbe::vlma(data_1.ptr(), rhs.arg2().get(0), rhs.arg3().get(0), 
      data_lhs.ptr(), lhs.size());
  }
};



} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

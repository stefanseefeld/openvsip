/* Copyright (c) 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/simd/eval-generic.hpp
    @author  Jules Bergmann
    @date    2006-01-25
    @brief   VSIPL++ Library: Wrappers and traits to bridge with generic SIMD.
*/

#ifndef VSIP_IMPL_SIMD_EVAL_GENERIC_HPP
#define VSIP_IMPL_SIMD_EVAL_GENERIC_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/dda.hpp>

#include <vsip/opt/simd/simd.hpp>
#include <vsip/opt/simd/vadd.hpp>
#include <vsip/opt/simd/vmul.hpp>
#include <vsip/opt/simd/svmul.hpp>
#include <vsip/opt/simd/vgt.hpp>
#include <vsip/opt/simd/vlogic.hpp>
#include <vsip/opt/simd/threshold.hpp>
#include <vsip/opt/simd/expr_iterator.hpp>

namespace vsip
{
namespace impl
{
namespace simd
{
template <template <typename, typename> class Operator>
struct Map_operator_to_algorithm
{
  typedef Alg_none type;
};

template <>
struct Map_operator_to_algorithm<expr::op::Add>  { typedef Alg_vadd type;};
template <>
struct Map_operator_to_algorithm<expr::op::Mult> { typedef Alg_vmul type;};
template <>
struct Map_operator_to_algorithm<expr::op::Band> { typedef Alg_vband type;};
template <>
struct Map_operator_to_algorithm<expr::op::Bor> { typedef Alg_vbor type;};
template <>
struct Map_operator_to_algorithm<expr::op::Bxor> { typedef Alg_vbxor type;};

} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

#define V_EXPR(OP, ALG, FCN)						\
template <typename LHS, typename LBlock>				\
struct Evaluator<op::assign<1>, be::simd_builtin,			\
		 void(LHS &, expr::Unary<OP, LBlock, true> const &)>	\
{									\
  typedef expr::Unary<OP, LBlock, true>  RHS;			        \
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LHS>::type>::type			\
    lhs_lp;								\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LBlock>::type>::type			\
    lblock_lp;								\
  									\
  static char const* name() { return "Expr_SIMD_V-" #FCN; }		\
  									\
  static bool const ct_valid = 						\
    !impl::is_expr_block<LBlock>::value &&				\
    impl::simd::is_algorithm_supported<					\
        typename LHS::value_type,					\
        impl::is_split_block<LHS>::value,				\
	ALG>::value &&							\
     /* check that direct access is supported */			\
    dda::Data<LHS, dda::out>::ct_cost == 0 &&				\
    dda::Data<LBlock, dda::in>::ct_cost == 0 &&				\
     /* Must have same complex interleaved/split format */		\
    get_block_layout<LHS>::storage_format ==				\
    get_block_layout<LBlock>::storage_format;				\
  									\
  static bool rt_valid(LHS &lhs, RHS const &rhs)		        \
  {									\
    /* check if all data is unit stride */				\
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);			\
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg());		\
    return (data_lhs.stride(0) == 1 && data_l.stride(0) == 1);		\
  }									\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);			\
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg());		\
    FCN(data_l.ptr(), data_lhs.ptr(), lhs.size());			\
  }									\
};

#define VV_EXPR(OP, FCN)						\
template <typename LHS, typename LBlock, typename RBlock>		\
struct Evaluator<op::assign<1>, be::simd_builtin,			\
		 void(LHS &,                                            \
		      expr::Binary<OP, LBlock, RBlock, true> const &)>	\
{									\
  typedef expr::Binary<OP, LBlock, RBlock, true> RHS;		        \
  									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LHS>::type>::type			\
    lhs_lp;								\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LBlock>::type>::type			\
    lblock_lp;								\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<RBlock>::type>::type			\
    rblock_lp;								\
                                                                        \
  static char const* name() { return "Expr_SIMD_VV-" #FCN; }		\
  									\
  static bool const ct_valid = 						\
    !impl::is_expr_block<LBlock>::value &&				\
    !impl::is_expr_block<RBlock>::value &&				\
    /* Check that input types of the Binary expression are the same */  \
    is_same<typename LBlock::value_type,				\
	    typename RBlock::value_type>::value &&			\
    /* Check that LHS & RHS have same type. */                          \
    is_same<typename LHS::value_type,					\
	    typename RHS::value_type>::value &&				\
    impl::simd::is_algorithm_supported<					\
      typename LHS::value_type,						\
      impl::is_split_block<LHS>::value,			  	\
      typename impl::simd::Map_operator_to_algorithm<OP>::type>::value && \
    /* check that direct access is supported */				\
    dda::Data<LHS, dda::out>::ct_cost == 0 &&				\
    dda::Data<LBlock, dda::in>::ct_cost == 0 &&				\
    dda::Data<RBlock, dda::in>::ct_cost == 0 &&				\
    /* Must have same complex interleaved/split format */		\
    get_block_layout<LHS>::storage_format ==				\
    get_block_layout<LBlock>::storage_format &&				\
    get_block_layout<LHS>::storage_format ==				\
    get_block_layout<RBlock>::storage_format;				\
  									\
  static bool rt_valid(LHS &lhs, RHS const &rhs)		        \
  {									\
    /* check if all data is unit stride */				\
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);			\
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg1());		\
    dda::Data<RBlock, dda::in, rblock_lp> data_r(rhs.arg2());		\
    return (data_lhs.stride(0) == 1 &&					\
	    data_l.stride(0) == 1 &&					\
	    data_r.stride(0) == 1);					\
  }									\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    dda::Data<LHS, dda::out, lhs_lp>  data_lhs(lhs);			\
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg1());		\
    dda::Data<RBlock, dda::in, rblock_lp> data_r(rhs.arg2());		\
    FCN(data_l.ptr(), data_r.ptr(), data_lhs.ptr(), lhs.size());	\
  }									\
};


V_EXPR (expr::op::Bnot, impl::simd::Alg_vbnot, impl::simd::vbnot)

#undef V_EXPR

VV_EXPR(expr::op::Mult, impl::simd::vmul)
VV_EXPR(expr::op::Add, impl::simd::vadd)
VV_EXPR(expr::op::Band, impl::simd::vband)
VV_EXPR(expr::op::Bor, impl::simd::vbor)
VV_EXPR(expr::op::Bxor, impl::simd::vbxor)

#undef VV_EXPR

/// Map 'A > B' to  vgt(A, B)
template <typename LHS, typename LBlock, typename RBlock>
struct Evaluator<op::assign<1>, be::simd_builtin,
  void(LHS &, expr::Binary<expr::op::Gt, LBlock, RBlock, true> const &)>
{
  typedef expr::Binary<expr::op::Gt, LBlock, RBlock, true> RHS;

  typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<LHS>::type>::type
    lhs_lp;
  typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<LBlock>::type>::type
    lblock_lp;
  typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<RBlock>::type>::type
    rblock_lp;

  static char const* name() { return "Expr_SIMD_VV-simd::vgt"; }

  static bool const ct_valid = 
    !impl::is_expr_block<LBlock>::value &&
    !impl::is_expr_block<RBlock>::value &&
    is_same<typename LHS::value_type, bool>::value &&
    is_same<typename RHS::value_type, bool>::value &&
    impl::simd::is_algorithm_supported<bool, false, impl::simd::Alg_vgt>::value &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<LBlock, dda::in>::ct_cost == 0 &&
    dda::Data<RBlock, dda::in>::ct_cost == 0;

  
  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_lp>  data_lhs(lhs);
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg1());
    dda::Data<RBlock, dda::in, rblock_lp> data_r(rhs.arg2());
    return (data_lhs.stride(0) == 1 &&
	    data_l.stride(0) == 1 &&
	    data_r.stride(0) == 1);
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg1());
    dda::Data<RBlock, dda::in, rblock_lp> data_r(rhs.arg2());
    impl::simd::vgt(data_l.ptr(), data_r.ptr(), data_lhs.ptr(), lhs.size());
  }
};

/// Map 'A < B' to  vgt(B, A)
template <typename LHS, typename LBlock, typename RBlock>
struct Evaluator<op::assign<1>, be::simd_builtin,
  void(LHS &, expr::Binary<expr::op::Lt, LBlock, RBlock, true> const &)>
{
  typedef expr::Binary<expr::op::Lt, LBlock, RBlock, true> RHS;

  typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<LHS>::type>::type
    lhs_lp;
  typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<LBlock>::type>::type
    lblock_lp;
  typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<RBlock>::type>::type
    rblock_lp;

  static char const* name() { return "Expr_SIMD_VV-simd::vlt_as_gt"; }

  static bool const ct_valid = 
    !impl::is_expr_block<LBlock>::value &&
    !impl::is_expr_block<RBlock>::value &&
    is_same<typename LHS::value_type, bool>::value &&
    is_same<typename LBlock::value_type, bool>::value &&
    is_same<typename RBlock::value_type, bool>::value &&
    impl::simd::is_algorithm_supported<bool, false, impl::simd::Alg_vgt>::value &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<LBlock, dda::in>::ct_cost == 0 &&
    dda::Data<RBlock, dda::in>::ct_cost == 0;

  
  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg1());
    dda::Data<RBlock, dda::in, rblock_lp> data_r(rhs.arg2());
    return (data_lhs.stride(0) == 1 &&
	    data_l.stride(0) == 1 &&
	    data_r.stride(0) == 1);
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg1());
    dda::Data<RBlock, dda::in, rblock_lp> data_r(rhs.arg2());
    // Swap left and right arguments to vgt
    impl::simd::vgt(data_r.ptr(), data_l.ptr(), data_lhs.ptr(), lhs.size());
  }
};

/// vector logical operators
#define LOGIC_V_EXPR(OP, ALG, FCN)			                \
template <typename LHS, typename Block>					\
struct Evaluator<op::assign<1>, be::simd_builtin,		        \
		 void(LHS &, expr::Unary<OP, Block, true> const &)>	\
{									\
  typedef expr::Unary<OP, Block, true> RHS;			        \
									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LHS>::type>::type			\
    lhs_lp;								\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<Block>::type>::type			\
    block_lp;								\
									\
  static char const* name() { return "Expr_SIMD_V-" #FCN; }		\
  									\
  static bool const ct_valid = 				   	        \
    !impl::is_expr_block<Block>::value &&				\
    is_same<typename LHS::value_type, bool>::value &&		\
    is_same<typename Block::value_type, bool>::value &&	\
    impl::simd::is_algorithm_supported<bool, false, ALG>::value &&	\
    /* check that direct access is supported */				\
    dda::Data<LHS, dda::out>::ct_cost == 0 &&				\
    dda::Data<Block, dda::in>::ct_cost == 0;				\
									\
  static bool rt_valid(LHS &lhs, RHS const &rhs)		        \
  {									\
    /* check if all data is unit stride */				\
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);			\
    dda::Data<Block, dda::in, block_lp> data_l(rhs.arg());		\
    return (data_lhs.stride(0) == 1 && data_l.stride(0) == 1);		\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);			\
    dda::Data<Block, dda::in, block_lp> data_l(rhs.arg());		\
    FCN(data_l.ptr(), data_lhs.ptr(), lhs.size());			\
  }									\
};

#define LOGIC_VV_EXPR(OP, ALG, FCN)			                \
template <typename LHS, typename LBlock, typename RBlock>		\
struct Evaluator<op::assign<1>, be::simd_builtin,			\
  void(LHS &, expr::Binary<OP, LBlock, RBlock, true> const &)>	        \
{									\
  typedef expr::Binary<OP, LBlock, RBlock, true> RHS;		        \
									\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LHS>::type>::type			\
    lhs_lp;								\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<LBlock>::type>::type			\
    lblock_lp;								\
  typedef typename impl::adjust_layout_dim<				\
  1, typename get_block_layout<RBlock>::type>::type			\
    rblock_lp;								\
									\
  static char const* name() { return "Expr_SIMD_VV-" #FCN; }		\
  									\
  static bool const ct_valid = 						\
    !impl::is_expr_block<LBlock>::value &&				\
    !impl::is_expr_block<RBlock>::value &&				\
    is_same<typename LHS::value_type, bool>::value &&		\
    is_same<typename LBlock::value_type, bool>::value &&	\
    is_same<typename RBlock::value_type, bool>::value &&	\
    impl::simd::is_algorithm_supported<bool, false, ALG>::value &&	\
    /* check that direct access is supported */				\
    dda::Data<LHS, dda::out>::ct_cost == 0 &&				\
    dda::Data<LBlock, dda::in>::ct_cost == 0 &&				\
    dda::Data<RBlock, dda::in>::ct_cost == 0;				\
									\
  static bool rt_valid(LHS &lhs, RHS const &rhs)		        \
  {									\
    /* check if all data is unit stride */				\
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);			\
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg1());		\
    dda::Data<RBlock, dda::in, rblock_lp> data_r(rhs.arg2());		\
    return (data_lhs.stride(0) == 1 &&					\
	    data_l.stride(0) == 1 &&					\
	    data_r.stride(0) == 1);					\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);			\
    dda::Data<LBlock, dda::in, lblock_lp> data_l(rhs.arg1());		\
    dda::Data<RBlock, dda::in, rblock_lp> data_r(rhs.arg2());		\
    FCN(data_l.ptr(), data_r.ptr(), data_lhs.ptr(), lhs.size());	\
  }									\
};

LOGIC_V_EXPR(expr::op::Lnot, impl::simd::Alg_vlnot, impl::simd::vlnot)
LOGIC_VV_EXPR(expr::op::Land, impl::simd::Alg_vland, impl::simd::vland)
LOGIC_VV_EXPR(expr::op::Lor,  impl::simd::Alg_vlor,  impl::simd::vlor)
LOGIC_VV_EXPR(expr::op::Lxor, impl::simd::Alg_vlxor, impl::simd::vlxor)

#undef LOGIC_V_EXPR
#undef LOGIC_VV_EXPR


/// Evaluate scalar * view
template <typename LHS, typename T, typename B>
struct Evaluator<op::assign<1>, be::simd_builtin,
  void(LHS &, expr::Binary<expr::op::Mult, expr::Scalar<1, T> const, B, true> const &)>
{
  typedef expr::Binary<expr::op::Mult, expr::Scalar<1, T> const, B, true> RHS;

  typedef typename impl::scalar_of<T>::type op1_scalar_type;
  typedef typename impl::scalar_of<typename B::value_type>::type op2_scalar_type;
  typedef typename impl::scalar_of<typename LHS::value_type>::type lhs_scalar_type;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<LHS>::type>::type
  lhs_lp;
  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<B>::type>::type
  vblock_lp;

  static char const* name() { return "Expr_SIMD_V-simd::svmul"; }

  static bool const ct_valid = 
    !impl::is_expr_block<B>::value &&
    impl::simd::is_algorithm_supported<
        T,
        impl::is_split_block<LHS>::value,
    typename impl::simd::Map_operator_to_algorithm<expr::op::Mult>::type>::value &&
    // all values must have the same precision
    is_same<op1_scalar_type, op2_scalar_type>::value &&
    is_same<op1_scalar_type, lhs_scalar_type>::value &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<B, dda::in>::ct_cost == 0 &&
    // Must have same complex interleaved/split format
    get_block_layout<LHS>::storage_format == get_block_layout<B>::storage_format;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);
    dda::Data<B, dda::in, vblock_lp> data_r(rhs.arg2());
    return (data_lhs.stride(0) == 1 && data_r.stride(0) == 1);
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);
    dda::Data<B, dda::in, vblock_lp> data_r(rhs.arg2());
    impl::simd::svmul(rhs.arg1().value(), data_r.ptr(), data_lhs.ptr(), lhs.size());
  }
};



/// Evaluate view * scalar
template <typename LHS, typename T, typename B>
struct Evaluator<op::assign<1>, be::simd_builtin,
  void(LHS &, expr::Binary<expr::op::Mult, B, expr::Scalar<1, T> const, true> const &)>
{
  typedef expr::Binary<expr::op::Mult, B, expr::Scalar<1, T> const, true> RHS;

  typedef typename impl::scalar_of<typename B::value_type>::type op1_scalar_type;
  typedef typename impl::scalar_of<T>::type op2_scalar_type;
  typedef typename impl::scalar_of<typename LHS::value_type>::type lhs_scalar_type;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<LHS>::type>::type
  lhs_lp;
  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<B>::type>::type
  vblock_lp;

  static char const* name() { return "Expr_SIMD_V-simd::svmul"; }

  static bool const ct_valid = 
    !impl::is_expr_block<B>::value &&
    impl::simd::is_algorithm_supported<
        T,
        impl::is_split_block<LHS>::value,
    typename impl::simd::Map_operator_to_algorithm<expr::op::Mult>::type>::value &&
    // all values must have the same precision
    is_same<op1_scalar_type, op2_scalar_type>::value &&
    is_same<op1_scalar_type, lhs_scalar_type>::value &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<B, dda::in>::ct_cost == 0 &&
    // Must have same complex interleaved/split format
    get_block_layout<LHS>::storage_format == get_block_layout<B>::storage_format;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);
    dda::Data<B, dda::in, vblock_lp> data_l(rhs.arg1());
    return (data_lhs.stride(0) == 1 && data_l.stride(0) == 1);
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);
    dda::Data<B, dda::in, vblock_lp> data_l(rhs.arg1());
    impl::simd::svmul(rhs.arg2().value(), data_l.ptr(), data_lhs.ptr(), lhs.size());
  }
};

///   threshold: vector threshold operator
///   ite(A > B, A, k)
///   ite(A < B, A, k)
///   ite(A <= B, A, k)
///   ite(A >= B, A, k)
template <typename LHS,
	  template <typename,typename> class O,
	  typename Block1,
	  typename Block2,
          typename T>
struct Evaluator<op::assign<1>, be::simd_builtin,
		 void(LHS &,
		      expr::Ternary<expr::op::Ite,
		        expr::Binary<O, Block1, Block2, true> const,
		        Block1,
		        expr::Scalar<1, T> const, 
		        true> const &)>
{

  typedef expr::Ternary<expr::op::Ite,
			expr::Binary<O, Block1, Block2, true> const,
			Block1,
			expr::Scalar<1, T> const,
			true> RHS;

  static char const* name() { return "Expr_SIMD_threshold"; }

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<LHS>::type>::type
  lhs_lp;
  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<Block1>::type>::type
  a_lp;
  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<Block2>::type>::type
  b_lp;

  static bool const ct_valid = 
    // Check that LHS & RHS have same type.
    is_same<typename LHS::value_type, T>::value &&
    // Make sure algorithm/op is supported.
    impl::simd::is_algorithm_supported<T, false, impl::simd::Alg_threshold>::value &&
    impl::simd::Binary_operator_map<T,O>::is_supported &&
    // Check that direct access is supported.
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;
  
  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    typedef impl::simd::Simd_traits<typename RHS::value_type> traits;

    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);
    dda::Data<Block1, dda::in, a_lp> data_a(rhs.arg1().arg1());
    dda::Data<Block2, dda::in, b_lp> data_b(rhs.arg1().arg2());
    return(data_lhs.stride(0) == 1 &&
           data_a.stride(0) == 1 &&
	   data_b.stride(0) == 1 &&
	   // make sure (A op B, A, k)
	   (&(rhs.arg1().arg1()) == &(rhs.arg2())) &&
	   // make sure everyting has same alignment
           (traits::alignment_of(data_lhs.ptr()) ==
	    traits::alignment_of(data_a.ptr())) &&
           (traits::alignment_of(data_lhs.ptr()) ==
	    traits::alignment_of(data_b.ptr())));
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    dda::Data<LHS, dda::out, lhs_lp> data_lhs(lhs);
    dda::Data<Block1, dda::in, a_lp> data_a(rhs.arg1().arg1());
    dda::Data<Block2, dda::in, b_lp> data_b(rhs.arg1().arg2());

    impl::simd::threshold<O>(data_lhs.ptr(), data_a.ptr(), data_b.ptr(),
			     rhs.arg3().value(), data_lhs.size());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_IMPL_SIMD_EVAL_GENERIC_HPP

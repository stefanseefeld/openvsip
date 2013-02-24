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
#include <vsip/core/extdata.hpp>

#include <vsip/opt/simd/simd.hpp>
#include <vsip/opt/simd/vadd.hpp>
#include <vsip/opt/simd/vmul.hpp>
#include <vsip/opt/simd/rscvmul.hpp>
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
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<LHS>::layout_type>::type		\
    dst_lp;								\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<LBlock>::layout_type>::type		\
    lblock_lp;								\
  									\
  static char const* name() { return "Expr_SIMD_V-" #FCN; }		\
  									\
  static bool const ct_valid = 						\
    !impl::Is_expr_block<LBlock>::value &&				\
    impl::simd::Is_algorithm_supported<					\
        typename LHS::value_type,					\
        impl::Is_split_block<LHS>::value,				\
	ALG>::value &&							\
     /* check that direct access is supported */			\
     impl::Ext_data_cost<LHS>::value == 0 &&				\
     impl::Ext_data_cost<LBlock>::value == 0 &&				\
     /* Must have same complex interleaved/split format */		\
     impl::Type_equal<typename impl::Block_layout<LHS>::complex_type,	\
		typename impl::Block_layout<LBlock>::complex_type>::value;\
  									\
  static bool rt_valid(LHS &lhs, RHS const &rhs)		        \
  {									\
    /* check if all data is unit stride */				\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);		\
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg(), impl::SYNC_IN);	\
    return (ext_dst.stride(0) == 1 && ext_l.stride(0) == 1);		\
  }									\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);		\
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg(), impl::SYNC_IN);	\
    FCN(ext_l.data(), ext_dst.data(), lhs.size());			\
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
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    dst_lp;								\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LBlock>::layout_type>::type	\
    lblock_lp;								\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<RBlock>::layout_type>::type	\
    rblock_lp;								\
  									\
  static char const* name() { return "Expr_SIMD_VV-" #FCN; }		\
  									\
  static bool const ct_valid = 						\
    !impl::Is_expr_block<LBlock>::value &&				\
    !impl::Is_expr_block<RBlock>::value &&				\
    impl::Type_equal<typename LHS::value_type, bool>::value &&	        \
    impl::simd::Is_algorithm_supported<					\
        typename LHS::value_type,					\
        impl::Is_split_block<LHS>::value,			  	\
	typename impl::simd::Map_operator_to_algorithm<OP>::type>::value &&\
     /* check that direct access is supported */			\
     impl::Ext_data_cost<LHS>::value == 0 &&				\
     impl::Ext_data_cost<LBlock>::value == 0 &&				\
     impl::Ext_data_cost<RBlock>::value == 0 &&				\
     /* Must have same complex interleaved/split format */		\
     impl::Type_equal<typename impl::Block_layout<LHS>::complex_type,	\
		typename impl::Block_layout<LBlock>::complex_type>::value &&\
     impl::Type_equal<typename impl::Block_layout<LHS>::complex_type,	\
		typename impl::Block_layout<RBlock>::complex_type>::value;\
  									\
  static bool rt_valid(LHS &lhs, RHS const &rhs)		        \
  {									\
    /* check if all data is unit stride */				\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);		\
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);	\
    impl::Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);	\
    return (ext_dst.stride(0) == 1 &&					\
	    ext_l.stride(0) == 1 &&					\
	    ext_r.stride(0) == 1);					\
  }									\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp>  ext_dst(lhs, impl::SYNC_OUT);		\
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);	\
    impl::Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);	\
    FCN(ext_l.data(), ext_r.data(), ext_dst.data(), lhs.size());	\
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

  typedef typename impl::Adjust_layout_dim<
      1, typename impl::Block_layout<LHS>::layout_type>::type
    dst_lp;
  typedef typename impl::Adjust_layout_dim<
      1, typename impl::Block_layout<LBlock>::layout_type>::type
    lblock_lp;
  typedef typename impl::Adjust_layout_dim<
      1, typename impl::Block_layout<RBlock>::layout_type>::type
    rblock_lp;

  static char const* name() { return "Expr_SIMD_VV-simd::vgt"; }

  static bool const ct_valid = 
    !impl::Is_expr_block<LBlock>::value &&
    !impl::Is_expr_block<RBlock>::value &&
    impl::Type_equal<typename LHS::value_type, bool>::value &&
    impl::Type_equal<typename RHS::value_type, bool>::value &&
    impl::simd::Is_algorithm_supported<bool, false, impl::simd::Alg_vgt>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<LHS>::value == 0 &&
    impl::Ext_data_cost<LBlock>::value == 0 &&
    impl::Ext_data_cost<RBlock>::value == 0;

  
  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    impl::Ext_data<LHS, dst_lp>  ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);
    impl::Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);
    return (ext_dst.stride(0) == 1 &&
	    ext_l.stride(0) == 1 &&
	    ext_r.stride(0) == 1);
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);
    impl::Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);
    impl::simd::vgt(ext_l.data(), ext_r.data(), ext_dst.data(), lhs.size());
  }
};

/// Map 'A < B' to  vgt(B, A)
template <typename LHS, typename LBlock, typename RBlock>
struct Evaluator<op::assign<1>, be::simd_builtin,
  void(LHS &, expr::Binary<expr::op::Lt, LBlock, RBlock, true> const &)>
{
  typedef expr::Binary<expr::op::Lt, LBlock, RBlock, true> RHS;

  typedef typename impl::Adjust_layout_dim<
      1, typename impl::Block_layout<LHS>::layout_type>::type
    dst_lp;
  typedef typename impl::Adjust_layout_dim<
      1, typename impl::Block_layout<LBlock>::layout_type>::type
    lblock_lp;
  typedef typename impl::Adjust_layout_dim<
      1, typename impl::Block_layout<RBlock>::layout_type>::type
    rblock_lp;

  static char const* name() { return "Expr_SIMD_VV-simd::vlt_as_gt"; }

  static bool const ct_valid = 
    !impl::Is_expr_block<LBlock>::value &&
    !impl::Is_expr_block<RBlock>::value &&
    impl::Type_equal<typename LHS::value_type, bool>::value &&
    impl::Type_equal<typename LBlock::value_type, bool>::value &&
    impl::Type_equal<typename RBlock::value_type, bool>::value &&
    impl::simd::Is_algorithm_supported<bool, false, impl::simd::Alg_vgt>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<LHS>::value == 0 &&
    impl::Ext_data_cost<LBlock>::value == 0 &&
    impl::Ext_data_cost<RBlock>::value == 0;

  
  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);
    impl::Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);
    return (ext_dst.stride(0) == 1 &&
	    ext_l.stride(0) == 1 &&
	    ext_r.stride(0) == 1);
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);
    impl::Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);
    // Swap left and right arguments to vgt
    impl::simd::vgt(ext_r.data(), ext_l.data(), ext_dst.data(), lhs.size());
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
  typedef typename impl::Adjust_layout_dim<				\
    1, typename impl::Block_layout<LHS>::layout_type>::type		\
    dst_lp;								\
  typedef typename impl::Adjust_layout_dim<				\
    1, typename impl::Block_layout<Block>::layout_type>::type		\
    block_lp;								\
									\
  static char const* name() { return "Expr_SIMD_V-" #FCN; }		\
  									\
  static bool const ct_valid = 				   	        \
    !impl::Is_expr_block<Block>::value &&				\
    impl::Type_equal<typename LHS::value_type, bool>::value &&		\
    impl::Type_equal<typename Block::value_type, bool>::value &&	\
    impl::simd::Is_algorithm_supported<bool, false, ALG>::value &&	\
    /* check that direct access is supported */				\
    impl::Ext_data_cost<LHS>::value == 0 &&				\
    impl::Ext_data_cost<Block>::value == 0;				\
									\
  static bool rt_valid(LHS &lhs, RHS const &rhs)		        \
  {									\
    /* check if all data is unit stride */				\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);		\
    impl::Ext_data<Block, block_lp> ext_l(rhs.arg(), impl::SYNC_IN);	\
    return (ext_dst.stride(0) == 1 && ext_l.stride(0) == 1);		\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);		\
    impl::Ext_data<Block, block_lp> ext_l(rhs.arg(),  impl::SYNC_IN);	\
    FCN(ext_l.data(), ext_dst.data(), lhs.size());			\
  }									\
};

#define LOGIC_VV_EXPR(OP, ALG, FCN)			                \
template <typename LHS, typename LBlock, typename RBlock>		\
struct Evaluator<op::assign<1>, be::simd_builtin,			\
  void(LHS &, expr::Binary<OP, LBlock, RBlock, true> const &)>	        \
{									\
  typedef expr::Binary<OP, LBlock, RBlock, true> RHS;		        \
									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    dst_lp;								\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LBlock>::layout_type>::type	\
    lblock_lp;								\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<RBlock>::layout_type>::type	\
    rblock_lp;								\
									\
  static char const* name() { return "Expr_SIMD_VV-" #FCN; }		\
  									\
  static bool const ct_valid = 						\
    !impl::Is_expr_block<LBlock>::value &&				\
    !impl::Is_expr_block<RBlock>::value &&				\
    impl::Type_equal<typename LHS::value_type, bool>::value &&		\
    impl::Type_equal<typename LBlock::value_type, bool>::value &&	\
    impl::Type_equal<typename RBlock::value_type, bool>::value &&	\
    impl::simd::Is_algorithm_supported<bool, false, ALG>::value &&	\
    /* check that direct access is supported */				\
    impl::Ext_data_cost<LHS>::value == 0 &&				\
    impl::Ext_data_cost<LBlock>::value == 0 &&				\
    impl::Ext_data_cost<RBlock>::value == 0;				\
									\
  static bool rt_valid(LHS &lhs, RHS const &rhs)		        \
  {									\
    /* check if all data is unit stride */				\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);		\
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);	\
    impl::Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);	\
    return (ext_dst.stride(0) == 1 &&					\
	    ext_l.stride(0) == 1 &&					\
	    ext_r.stride(0) == 1);					\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);		\
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);	\
    impl::Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);	\
    FCN(ext_l.data(), ext_r.data(), ext_dst.data(), lhs.size());	\
  }									\
};

LOGIC_V_EXPR(expr::op::Lnot, impl::simd::Alg_vlnot, impl::simd::vlnot)
LOGIC_VV_EXPR(expr::op::Land, impl::simd::Alg_vland, impl::simd::vland)
LOGIC_VV_EXPR(expr::op::Lor,  impl::simd::Alg_vlor,  impl::simd::vlor)
LOGIC_VV_EXPR(expr::op::Lxor, impl::simd::Alg_vlxor, impl::simd::vlxor)

#undef LOGIC_V_EXPR
#undef LOGIC_VV_EXPR


/// Evaluate real-scalar * complex-view
template <typename LHS, typename T, typename B>
struct Evaluator<op::assign<1>, be::simd_builtin,
  void(LHS &, expr::Binary<expr::op::Mult, expr::Scalar<1, T>, B, true> const &)>
{
  typedef expr::Binary<expr::op::Mult, expr::Scalar<1, T>, B, true> RHS;

  typedef typename impl::Adjust_layout_dim<
    1, typename impl::Block_layout<LHS>::layout_type>::type
  dst_lp;
  typedef typename impl::Adjust_layout_dim<
    1, typename impl::Block_layout<B>::layout_type>::type
  vblock_lp;

  static char const* name() { return "Expr_SIMD_V-simd::rscvmul"; }

  static bool const ct_valid = 
    !impl::Is_expr_block<B>::value &&
    impl::simd::Is_algorithm_supported<
        T,
        impl::Is_split_block<LHS>::value,
    typename impl::simd::Map_operator_to_algorithm<expr::op::Mult>::type>::value &&
    impl::Type_equal<typename LHS::value_type, std::complex<T> >::value &&
    // check that direct access is supported
    impl::Ext_data_cost<LHS>::value == 0 &&
    impl::Ext_data_cost<B>::value == 0 &&
    // Must have same complex interleaved/split format
    impl::Type_equal<typename impl::Block_layout<LHS>::complex_type,
		     typename impl::Block_layout<B>::complex_type>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<B, vblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);
    return (ext_dst.stride(0) == 1 && ext_r.stride(0) == 1);
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<B, vblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);
    impl::simd::rscvmul(rhs.arg1().value(), ext_r.data(), ext_dst.data(), lhs.size());
  }
};



/// Evaluate complex-view * real-scalar
template <typename LHS, typename T, typename B>
struct Evaluator<op::assign<1>, be::simd_builtin,
  void(LHS &, expr::Binary<expr::op::Mult, B, expr::Scalar<1, T>, true> const &)>
{
  typedef expr::Binary<expr::op::Mult, B, expr::Scalar<1, T>, true> RHS;

  typedef typename impl::Adjust_layout_dim<
    1, typename impl::Block_layout<LHS>::layout_type>::type
  dst_lp;
  typedef typename impl::Adjust_layout_dim<
    1, typename impl::Block_layout<B>::layout_type>::type
  vblock_lp;

  static char const* name() { return "Expr_SIMD_V-simd::rscvmul"; }

  static bool const ct_valid = 
    !impl::Is_expr_block<B>::value &&
    impl::simd::Is_algorithm_supported<
        T,
        impl::Is_split_block<LHS>::value,
    typename impl::simd::Map_operator_to_algorithm<expr::op::Mult>::type>::value &&
    impl::Type_equal<typename LHS::value_type, std::complex<T> >::value &&
    // check that direct access is supported
    impl::Ext_data_cost<LHS>::value == 0 &&
    impl::Ext_data_cost<B>::value == 0 &&
    // Must have same complex interleaved/split format
    impl::Type_equal<typename impl::Block_layout<LHS>::complex_type,
		     typename impl::Block_layout<B>::complex_type>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<B, vblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);
    return (ext_dst.stride(0) == 1 && ext_l.stride(0) == 1);
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<B, vblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);
    impl::simd::rscvmul(rhs.arg2().value(), ext_l.data(), ext_dst.data(), lhs.size());
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
		        expr::Scalar<1, T>, 
		        true> const &)>
{

  typedef expr::Ternary<expr::op::Ite,
			expr::Binary<O, Block1, Block2, true> const,
			Block1,
			expr::Scalar<1, T>,
			true> RHS;

  static char const* name() { return "Expr_SIMD_threshold"; }

  typedef typename impl::Adjust_layout_dim<
    1, typename impl::Block_layout<LHS>::layout_type>::type
  dst_lp;
  typedef typename impl::Adjust_layout_dim<
    1, typename impl::Block_layout<Block1>::layout_type>::type
  a_lp;
  typedef typename impl::Adjust_layout_dim<
    1, typename impl::Block_layout<Block2>::layout_type>::type
  b_lp;

  static bool const ct_valid = 
    // Check that LHS & RHS have same type.
    impl::Type_equal<typename LHS::value_type, T>::value &&
    // Make sure algorithm/op is supported.
    impl::simd::Is_algorithm_supported<T, false, impl::simd::Alg_threshold>::value &&
    impl::simd::Binary_operator_map<T,O>::is_supported &&
    // Check that direct access is supported.
    impl::Ext_data_cost<LHS>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0;
  
  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    typedef impl::simd::Simd_traits<typename RHS::value_type> traits;

    // check if all data is unit stride
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<Block1, a_lp> ext_a(rhs.arg1().arg1(), impl::SYNC_IN);
    impl::Ext_data<Block2, b_lp> ext_b(rhs.arg1().arg2(), impl::SYNC_IN);
    return(ext_dst.stride(0) == 1 &&
           ext_a.stride(0) == 1 &&
	   ext_b.stride(0) == 1 &&
	   // make sure (A op B, A, k)
	   (&(rhs.arg1().arg1()) == &(rhs.arg2())) &&
	   // make sure everyting has same alignment
           (traits::alignment_of(ext_dst.data()) ==
	    traits::alignment_of(ext_a.data())) &&
           (traits::alignment_of(ext_dst.data()) ==
	    traits::alignment_of(ext_b.data())));
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);
    impl::Ext_data<Block1, a_lp> ext_a(rhs.arg1().arg1(), impl::SYNC_IN);
    impl::Ext_data<Block2, b_lp> ext_b(rhs.arg1().arg2(), impl::SYNC_IN);

    impl::simd::threshold<O>(ext_dst.data(), ext_a.data(), ext_b.data(),
			     rhs.arg3().value(), ext_dst.size());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_IMPL_SIMD_EVAL_GENERIC_HPP

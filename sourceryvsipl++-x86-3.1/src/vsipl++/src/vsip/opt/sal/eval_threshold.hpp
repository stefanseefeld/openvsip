/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/eval_threshold.hpp
    @author  Jules Bergmann
    @date    2006-10-18
    @brief   VSIPL++ Library: Dispatch for Mercury SAL -- threshold.
*/

#ifndef VSIP_OPT_SAL_EVAL_THRESHOLD_HPP
#define VSIP_OPT_SAL_EVAL_THRESHOLD_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/opt/sal/eval_util.hpp>
#include <vsip/core/adjust_layout.hpp>

namespace vsip
{
namespace impl
{
namespace sal
{
// This definition (and the above ifdef) will go away once the code
// has benchmarked.
// VSIP_IMPL_SAL_GTE_THRESH_EXPR(float, sal::vthresh, sal::vthresh0)



// Common evaluator for a threshold expressions.

template <typename LHS, typename T, typename B>
struct Threshold_evaluator
{
  static char const* name() { return "Expr_SAL_thresh"; }

  typedef typename sal::Effective_value_type<LHS>::type eff_dst_t;
  typedef typename sal::Effective_value_type<B, T>::type eff_1_t;

  typedef typename adjust_layout_dim<
      1, typename get_block_layout<LHS>::type>::type
    dst_lp;
  
  typedef typename adjust_layout_dim<
      1, typename get_block_layout<B>::type>::type
    block1_lp;

  static bool const ct_valid =
    is_same<T, float>::value &&
    /* check that direct access is supported */
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<B, dda::in>::ct_cost == 0;

  static bool rt_valid(LHS&,
		       B const &a1,
		       expr::Scalar<1, T> const &b,
		       B const &a2,
		       expr::Scalar<1, T> const &c)
  {
    return &a1 == &a2 && (b.value() == c.value() || c.value() == T(0));
  }

  static void exec(LHS &lhs,
		   B const &a,
		   expr::Scalar<1, T> const &b,
		   B const &,
		   expr::Scalar<1, T> const &c)
  {
    typedef expr::Scalar<1, T> sb_type;

    sal::DDA_wrapper<LHS, dda::out, dst_lp>  ext_dst(lhs);
    sal::DDA_wrapper<B, dda::in, block1_lp> ext_A(a);
    sal::DDA_wrapper<sb_type, dda::in> ext_b(b);

    if (c.value() == T(0))
      sal::vthresh0(typename sal::DDA_wrapper<B, dda::in, block1_lp>::sal_type(ext_A),
		    typename sal::DDA_wrapper<sb_type, dda::in>::sal_type(ext_b),
		    typename sal::DDA_wrapper<LHS, dda::out, dst_lp>::sal_type(ext_dst),
		    lhs.size());
    else
      sal::vthresh(typename sal::DDA_wrapper<B, dda::in, block1_lp>::sal_type(ext_A),
		   typename sal::DDA_wrapper<sb_type, dda::in>::sal_type(ext_b),
		   typename sal::DDA_wrapper<LHS, dda::out, dst_lp>::sal_type(ext_dst),
		   lhs.size());
  }
};
} // namespace vsip::impl::sal
} // namespace vsip::impl
} // nmaespace vsip


namespace vsip_csl
{
namespace dispatcher
{

// Optimize threshold expression: ite(A > b, A, c)
#define VSIP_IMPL_SAL_GTE_THRESH_EXPR(T0, FUN, FUN0)			\
template <typename LHS, typename T, typename B>				\
struct Evaluator<op::assign<1>, be::mercury_sal,			\
  void(LHS &,               					\
       expr::Ternary<expr::op::Ite,      				\
                     expr::Binary<expr::op::Ge, B,                      \
                                  expr::Scalar<1, T>, true> const,      \
                     B, expr::Scalar<1, T>, true> const &)>		\
{									\
  static char const* name() { return "Expr_SAL_thresh"; }		\
									\
  typedef expr::Ternary<expr::op::Ite,  				\
    expr::Binary<expr::op::Ge, B, expr::Scalar<1, T>, true> const,      \
    B, expr::Scalar<1, T>, true>				        \
       RHS;							        \
									\
  typedef typename LHS::value_type lhs_value_type;			\
									\
  typedef typename impl::sal::Effective_value_type<LHS>::type eff_dst_t;\
  typedef typename impl::sal::Effective_value_type<B, T>::type eff_1_t;	\
									\
  typedef typename impl::adjust_layout_dim<				\
      1, typename get_block_layout<LHS>::type>::type		\
    dst_lp;								\
  									\
  typedef typename impl::adjust_layout_dim<				\
      1, typename get_block_layout<B>::type>::type		\
    block1_lp;								\
  									\
  static bool const ct_valid =						\
     is_same<T, T0>::value &&					\
     /* check that direct access is supported */			\
    dda::Data<LHS, dda::out>::ct_cost == 0 &&				\
    dda::Data<B, dda::in>::ct_cost == 0;					\
									\
  static bool rt_valid(LHS &, RHS const &rhs)   			\
  {									\
    return &(rhs.arg1().arg1()) == &(rhs.arg2()) &&			\
           (rhs.arg1().arg2().value() == rhs.arg3().value() ||	        \
            rhs.arg3().value() == T(0));				\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)     			\
  {									\
    using namespace impl;                                               \
    typedef expr::Scalar<1, T> sb_type;				        \
									\
    sal::DDA_wrapper<LHS, dda::out, dst_lp>  ext_dst(lhs);	        \
    sal::DDA_wrapper<B, dda::in, block1_lp> ext_A(rhs.arg2());	        \
    sal::DDA_wrapper<sb_type, dda::in> ext_b(rhs.arg1().arg2());        \
									\
    if (rhs.arg3().value() == T(0))					\
      FUN0(typename sal::DDA_wrapper<B, dda::in, block1_lp>::sal_type(ext_A),    \
           typename sal::DDA_wrapper<sb_type, dda::in>::sal_type(ext_b),		\
	   typename sal::DDA_wrapper<LHS, dda::out, dst_lp>::sal_type(ext_dst),   \
	   lhs.size());							\
    else								\
      FUN(typename sal::DDA_wrapper<B, dda::in, block1_lp>::sal_type(ext_A),     \
	  typename sal::DDA_wrapper<sb_type, dda::in>::sal_type(ext_b),		\
	  typename sal::DDA_wrapper<LHS, dda::out, dst_lp>::sal_type(ext_dst),    \
	  lhs.size());							\
  }									\
};

/// Frontend for threshold expressions like:
///
///   ite(A >= b, A, c)
template <typename LHS, typename T, typename B>
struct Evaluator<op::assign<1>, be::mercury_sal,
  void(LHS &, 
       expr::Ternary<expr::op::Ite,
         expr::Binary<expr::op::Ge, B, expr::Scalar<1, T>, true> const,
           B, expr::Scalar<1, T>, true> const &)>
  : impl::sal::Threshold_evaluator<LHS, T, B>
{
  typedef impl::sal::Threshold_evaluator<LHS, T, B> base_type;

  typedef 
  expr::Ternary<expr::op::Ite,
		expr::Binary<expr::op::Ge, B, expr::Scalar<1, T>, true> const,
		B, expr::Scalar<1, T>, true>
  RHS;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    return base_type::rt_valid(lhs,
			       rhs.arg1().arg1(),
			       rhs.arg1().arg2(),
			       rhs.arg2(),
			       rhs.arg3());
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    base_type::exec(lhs,
		    rhs.arg1().arg1(),
		    rhs.arg1().arg2(),
		    rhs.arg2(),
		    rhs.arg3());
  }
};

/// Frontend for threshold expressions like:
///
///   ite(A < b, c, A)
template <typename LHS, typename T, typename B>
struct Evaluator<op::assign<1>, be::mercury_sal,
  void(LHS &, 
       expr::Ternary<expr::op::Ite,
         expr::Binary<expr::op::Lt, B, expr::Scalar<1, T>, true> const,
           expr::Scalar<1, T>, B, true> const &)>
  : impl::sal::Threshold_evaluator<LHS, T, B>
{
  typedef impl::sal::Threshold_evaluator<LHS, T, B> base_type;

  typedef
  expr::Ternary<expr::op::Ite,
    expr::Binary<expr::op::Lt, B, expr::Scalar<1, T>, true> const,
      expr::Scalar<1, T>, B, true>
  RHS;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    return base_type::rt_valid(lhs,
			       rhs.arg1().arg1(),
			       rhs.arg1().arg2(),
			       rhs.arg3(),
			       rhs.arg2());
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    base_type::exec(lhs,
		    rhs.arg1().arg1(),
		    rhs.arg1().arg2(),
		    rhs.arg3(),
		    rhs.arg2());
  }
};

/// Frontend for threshold expressions like:
///
///   ite(b <= A, A, c)
template <typename LHS, typename T, typename B>
struct Evaluator<op::assign<1>, be::mercury_sal,
  void(LHS &, 
       expr::Ternary<expr::op::Ite,
         expr::Binary<expr::op::Le, expr::Scalar<1, T>, B, true> const,
           B, expr::Scalar<1, T>, true> const &)>
  : impl::sal::Threshold_evaluator<LHS, T, B>
{
  typedef impl::sal::Threshold_evaluator<LHS, T, B> base_type;

  typedef 
  expr::Ternary<expr::op::Ite,
    expr::Binary<expr::op::Le, expr::Scalar<1, T>, B, true> const,
		B, expr::Scalar<1, T>, true>
  RHS;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    return base_type::rt_valid(lhs,
			       rhs.arg1().arg2(),
			       rhs.arg1().arg1(),
			       rhs.arg2(),
			       rhs.arg3());
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    base_type::exec(lhs,
		    rhs.arg1().arg2(),
		    rhs.arg1().arg1(),
		    rhs.arg2(),
		    rhs.arg3());
  }
};

/// Frontend for threshold expressions like:
///
///   ite(b > A, c, A)
template <typename LHS, typename T, typename B>
struct Evaluator<op::assign<1>, be::mercury_sal,
  void(LHS &, 
       expr::Ternary<expr::op::Ite,
         expr::Binary<expr::op::Gt, expr::Scalar<1, T>, B, true> const,
           expr::Scalar<1, T>, B, true> const &)>
  : impl::sal::Threshold_evaluator<LHS, T, B>
{
  typedef impl::sal::Threshold_evaluator<LHS, T, B> base_type;

  typedef
  expr::Ternary<expr::op::Ite,
    expr::Binary<expr::op::Gt, expr::Scalar<1, T>, B, true> const,
      expr::Scalar<1, T>, B, true>
  RHS;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    return base_type::rt_valid(lhs,
			       rhs.arg1().arg2(),
			       rhs.arg1().arg1(),
			       rhs.arg3(),
			       rhs.arg2());
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    base_type::exec(lhs,
		    rhs.arg1().arg2(),
		    rhs.arg1().arg1(),
		    rhs.arg3(),
		    rhs.arg2());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_SAL_EVAL_THRESHOLD_HPP

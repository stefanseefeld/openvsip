/* Copyright (c) 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/eval_elementwise.hpp
    @author  Jules Bergmann
    @date    2006-05-26
    @brief   VSIPL++ Library: Dispatch for Mercury SAL.
*/

#ifndef VSIP_OPT_SAL_EVAL_ELEMENTWISE_HPP
#define VSIP_OPT_SAL_EVAL_ELEMENTWISE_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/core/adjust_layout.hpp>
#include <vsip/core/view_cast.hpp>
#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/sal/eval_util.hpp>
#include <vsip/opt/sal/is_op_supported.hpp>

namespace vsip
{
namespace impl
{
namespace sal
{

/// Evaluator base class for view-view expressions with mixed value types
/// I.e. complex<float> = float * complex<float>
template <typename LHS,
	  template <typename, typename> class Operator,
	  typename LBlock,
	  typename RBlock>
struct Evaluator_mixed
{
  typedef expr::Binary<Operator, LBlock, RBlock, true> RHS;

  typedef typename LHS::value_type lhs_value_type;

  typedef typename sal::Effective_value_type<LHS>::type d_eff_t;
  typedef typename sal::Effective_value_type<LBlock>::type l_eff_t;
  typedef typename sal::Effective_value_type<RBlock>::type r_eff_t;

  static bool const ct_valid = 
    (!Is_expr_block<LBlock>::value || Is_scalar_block<LBlock>::value) &&
    (!Is_expr_block<RBlock>::value || Is_scalar_block<RBlock>::value) &&
     Is_op2_supported<Operator, l_eff_t, r_eff_t, d_eff_t>::value &&
     // check that direct access is supported
     Ext_data_cost<LHS>::value == 0 &&
     (Ext_data_cost<LBlock>::value == 0 || Is_scalar_block<LBlock>::value) &&
     (Ext_data_cost<RBlock>::value == 0 || Is_scalar_block<RBlock>::value);
  
  static bool rt_valid(LHS &, RHS const &) 
  {
    // SAL supports all strides.
    return true;
  }
};

/// Same as above but requires unit strides.
template <typename LHS,
	  template <typename, typename> class Operator,
	  typename LBlock,
	  typename RBlock>
struct Evaluator_unitstride
{
  typedef expr::Binary<Operator, LBlock, RBlock, true> RHS;

  typedef typename LHS::value_type lhs_value_type;

  typedef typename sal::Effective_value_type<LHS>::type d_eff_t;
  typedef typename sal::Effective_value_type<LBlock>::type l_eff_t;
  typedef typename sal::Effective_value_type<RBlock>::type r_eff_t;

  static bool const ct_valid = 
    (!Is_expr_block<LBlock>::value || Is_scalar_block<LBlock>::value) &&
    (!Is_expr_block<RBlock>::value || Is_scalar_block<RBlock>::value) &&
     Is_op2_supported<Operator, l_eff_t, r_eff_t, d_eff_t>::value &&
     // check that direct access is supported
     Ext_data_cost<LHS>::value == 0 &&
     (Ext_data_cost<LBlock>::value == 0 || Is_scalar_block<LBlock>::value) &&
     (Ext_data_cost<RBlock>::value == 0 || Is_scalar_block<RBlock>::value);
  
  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    Ext_data<LHS> ext_lhs(lhs, SYNC_OUT);
    Ext_data<LBlock> ext_l(rhs.arg1(), SYNC_IN);
    Ext_data<RBlock> ext_r(rhs.arg2(), SYNC_IN);
    return (ext_lhs.stride(0) == 1 &&
            ext_l.stride(0) == 1 &&
            ext_r.stride(0) == 1); 
  }
};

} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

#define VSIP_IMPL_SAL_COPY_EXPR(OP, FUN)				\
template <typename LHS, typename RHS>					\
 struct Evaluator<op::assign<1>,                                        \
   typename impl::enable_if<impl::Is_leaf_block<RHS>, be::mercury_sal>::type, \
   void(LHS &, RHS const &)>				                \
{									\
  static char const* name() { return "Expr_SAL_COPY"; }			\
									\
  typedef typename LHS::value_type lhs_value_type;			\
									\
  typedef typename impl::sal::Effective_value_type<LHS>::type eff_lhs_t;\
  typedef typename impl::sal::Effective_value_type<RHS>::type eff_rhs_t;\
									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<LHS>::layout_type>::type		\
    lhs_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<RHS>::layout_type>::type		\
    rhs_lp;								\
  									\
  static bool const ct_valid = 						\
    (!impl::Is_expr_block<RHS>::value || impl::Is_scalar_block<RHS>::value) && \
    impl::sal::Is_op1_supported<OP, eff_rhs_t, eff_lhs_t>::value&&	\
     /* check that direct access is supported */			\
    impl::Ext_data_cost<LHS>::value == 0 &&				\
    (impl::Ext_data_cost<RHS>::value == 0 ||				\
     impl::Is_scalar_block<RHS>::value);				\
									\
  static bool rt_valid(LHS &, RHS const &)			        \
  {									\
    /* SAL supports all strides */					\
    return true;							\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::sal::Ext_wrapper<LHS, lhs_lp> ext_lhs(lhs, impl::SYNC_OUT);	\
    impl::sal::Ext_wrapper<RHS, rhs_lp> ext_rhs(rhs, impl::SYNC_IN);	\
    FUN(typename impl::sal::Ext_wrapper<RHS, rhs_lp>::sal_type(ext_rhs),\
	typename impl::sal::Ext_wrapper<LHS, lhs_lp>::sal_type(ext_lhs),\
	lhs.size());						        \
  }									\
};

VSIP_IMPL_SAL_COPY_EXPR(impl::sal::copy_token, vcopy)

#define VSIP_IMPL_SAL_V_EXPR(OP, FUN)	                 		\
template <typename LHS, typename Block>					\
struct Evaluator<op::assign<1>,                                         \
  typename impl::enable_if<impl::Is_leaf_block<Block>, be::mercury_sal>::type, \
  void(LHS &, expr::Unary<OP, Block, true> const &)>                    \
{									\
  static char const* name() { return "Expr_SAL_V"; }			\
									\
  typedef expr::Unary<OP, Block, true> RHS;     			\
									\
  typedef typename LHS::value_type lhs_value_type;			\
									\
  typedef typename impl::sal::Effective_value_type<LHS>::type eff_lhs_t;\
  typedef typename impl::sal::Effective_value_type<Block>::type eff_1_t;\
									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    lhs_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<Block>::layout_type>::type		\
    block1_lp;								\
  									\
  static bool const ct_valid = 						\
    (!impl::Is_expr_block<Block>::value ||                              \
     impl::Is_scalar_block<Block>::value) &&				\
    impl::sal::Is_op1_supported<OP, eff_1_t, eff_lhs_t>::value&&	\
    /* check that direct access is supported */				\
    impl::Ext_data_cost<LHS>::value == 0 &&				\
    (impl::Ext_data_cost<Block>::value == 0 ||                          \
     impl::Is_scalar_block<Block>::value);			       	\
									\
  static bool rt_valid(LHS &, RHS const &)			        \
  {									\
    /* SAL supports all strides */					\
    return true;							\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    using namespace impl;                                               \
    sal::Ext_wrapper<LHS, lhs_lp>  ext_lhs(lhs, SYNC_OUT);              \
    sal::Ext_wrapper<Block, block1_lp> ext_1(rhs.arg(), SYNC_IN);       \
    FUN(typename sal::Ext_wrapper<Block, block1_lp>::sal_type(ext_1),   \
	typename sal::Ext_wrapper<LHS, lhs_lp>::sal_type(ext_lhs),      \
	lhs.size());							\
  }									\
};

VSIP_IMPL_SAL_V_EXPR(expr::op::Magsq, impl::sal::vmagsq)
VSIP_IMPL_SAL_V_EXPR(expr::op::Mag, impl::sal::vmag)
VSIP_IMPL_SAL_V_EXPR(expr::op::Minus, impl::sal::vneg)
VSIP_IMPL_SAL_V_EXPR(expr::op::Cos, impl::sal::vcos)
VSIP_IMPL_SAL_V_EXPR(expr::op::Sin, impl::sal::vsin)
VSIP_IMPL_SAL_V_EXPR(expr::op::Tan, impl::sal::vtan)
VSIP_IMPL_SAL_V_EXPR(expr::op::Atan, impl::sal::vatan)
VSIP_IMPL_SAL_V_EXPR(expr::op::Log, impl::sal::vlog)
VSIP_IMPL_SAL_V_EXPR(expr::op::Log10, impl::sal::vlog10)
VSIP_IMPL_SAL_V_EXPR(expr::op::Exp, impl::sal::vexp)
VSIP_IMPL_SAL_V_EXPR(expr::op::Exp10, impl::sal::vexp10)
VSIP_IMPL_SAL_V_EXPR(expr::op::Sqrt, impl::sal::vsqrt)
VSIP_IMPL_SAL_V_EXPR(expr::op::Rsqrt, impl::sal::vrsqrt)
VSIP_IMPL_SAL_V_EXPR(expr::op::Sq, impl::sal::vsq)
VSIP_IMPL_SAL_V_EXPR(expr::op::Recip, impl::sal::vrecip)

VSIP_IMPL_SAL_V_EXPR(impl::Cast_closure<long          >::Cast, impl::sal::vconv)
VSIP_IMPL_SAL_V_EXPR(impl::Cast_closure<short         >::Cast, impl::sal::vconv)
#if VSIP_IMPL_SAL_USES_SIGNED == 1
VSIP_IMPL_SAL_V_EXPR(impl::Cast_closure<signed char   >::Cast, impl::sal::vconv)
#else
VSIP_IMPL_SAL_V_EXPR(impl::Cast_closure<char          >::Cast, impl::sal::vconv)
#endif
VSIP_IMPL_SAL_V_EXPR(impl::Cast_closure<unsigned long >::Cast, impl::sal::vconv)
VSIP_IMPL_SAL_V_EXPR(impl::Cast_closure<unsigned short>::Cast, impl::sal::vconv)
VSIP_IMPL_SAL_V_EXPR(impl::Cast_closure<unsigned char >::Cast, impl::sal::vconv)

VSIP_IMPL_SAL_V_EXPR(impl::Cast_closure<float>::Cast, impl::sal::vconv)


#define VSIP_IMPL_SAL_VV_EXPR(OP, FUN)					\
template <typename LHS, typename LBlock, typename RBlock>		\
struct Evaluator<op::assign<1>,                                         \
  typename impl::enable_if_c<impl::Is_leaf_block<LBlock>::value &&      \
                             impl::Is_leaf_block<RBlock>::value,       	\
                             be::mercury_sal>::type,			\
  void(LHS &, expr::Binary<OP, LBlock, RBlock, true> const &)>	        \
  : impl::sal::Evaluator_mixed<LHS, OP, LBlock, RBlock>			\
{									\
  static char const* name() { return "Expr_SAL_VV"; }			\
									\
  typedef expr::Binary<OP, LBlock, RBlock, true> RHS;		        \
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    lhs_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LBlock>::layout_type>::type	\
    lblock_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<RBlock>::layout_type>::type	\
    rblock_lp;								\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::sal::Ext_wrapper<LHS, lhs_lp>  ext_lhs(lhs, impl::SYNC_OUT);  \
    impl::sal::Ext_wrapper<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);	\
    impl::sal::Ext_wrapper<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);\
									\
    assert(lhs.size() <= rhs.arg1().size() || rhs.arg1().size() == 0);	\
    assert(lhs.size() <= rhs.arg2().size() || rhs.arg2().size() == 0);  \
									\
    VSIP_IMPL_COVER_BLK("SAL_VV", RHS);			 	        \
    FUN(typename impl::sal::Ext_wrapper<LBlock, lblock_lp>::sal_type(ext_l),	\
        typename impl::sal::Ext_wrapper<RBlock, lblock_lp>::sal_type(ext_r),	\
        typename impl::sal::Ext_wrapper<LHS, lhs_lp>::sal_type(ext_lhs),	\
        lhs.size());							\
  }									\
};

#define VSIP_IMPL_SAL_VV_EXPR_UNIT_EXPR(OP, FUN)			\
template <typename LHS, typename LBlock, typename RBlock>		\
struct Evaluator<op::assign<1>,                                         \
  typename impl::enable_if_c<impl::Is_leaf_block<LBlock>::value &&      \
	 		     impl::Is_leaf_block<LBlock>::value,        \
			     be::mercury_sal>::type,			\
  void(LHS &, expr::Binary<OP, LBlock, RBlock, true> const &)>	        \
  : impl::sal::Evaluator_unitstride<LHS, OP, LBlock, RBlock>		\
{									\
  static char const* name() { return "Expr_SAL_VV"; }			\
									\
  typedef expr::Binary<OP, LBlock, RBlock, true> RHS;		        \
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    lhs_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<			      	\
      1, typename impl::Block_layout<LBlock>::layout_type>::type	\
    lblock_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<RBlock>::layout_type>::type	\
    rblock_lp;								\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    using namespace impl;                                               \
    sal::Ext_wrapper<LHS, lhs_lp> ext_lhs(lhs, SYNC_OUT);	        \
    sal::Ext_wrapper<LBlock, lblock_lp> ext_l(rhs.arg1(), SYNC_IN);	\
    sal::Ext_wrapper<RBlock, rblock_lp> ext_r(rhs.arg2(), SYNC_IN);	\
									\
    assert(lhs.size() <= rhs.arg1().size() || rhs.arg1().size() == 0);	\
    assert(lhs.size() <= rhs.arg2().size() || rhs.arg2().size() == 0);  \
									\
    VSIP_IMPL_COVER_BLK("SAL_VV", RHS);				        \
    FUN(typename sal::Ext_wrapper<LBlock, lblock_lp>::sal_type(ext_l),	\
        typename sal::Ext_wrapper<RBlock, lblock_lp>::sal_type(ext_r),	\
        typename sal::Ext_wrapper<LHS, lhs_lp>::sal_type(ext_lhs),	\
        lhs.size());							\
  }									\
};

VSIP_IMPL_SAL_VV_EXPR(expr::op::Add,  impl::sal::vadd)
VSIP_IMPL_SAL_VV_EXPR(expr::op::Sub,  impl::sal::vsub)
VSIP_IMPL_SAL_VV_EXPR(expr::op::Mult, impl::sal::vmul)
VSIP_IMPL_SAL_VV_EXPR(expr::op::Div,  impl::sal::vdiv)

VSIP_IMPL_SAL_VV_EXPR(expr::op::Max, impl::sal::vmax)
VSIP_IMPL_SAL_VV_EXPR(expr::op::Min, impl::sal::vmin)

VSIP_IMPL_SAL_VV_EXPR(expr::op::Band, impl::sal::vband)
VSIP_IMPL_SAL_VV_EXPR(expr::op::Bor,  impl::sal::vbor)

VSIP_IMPL_SAL_VV_EXPR_UNIT_EXPR(expr::op::Atan2,  impl::sal::vatan2)

#define VSIP_IMPL_SAL_VVV_EXPR(OP, FUN)					\
template <typename LHS, typename B1, typename B2, typename B3>          \
struct Evaluator<op::assign<1>, be::mercury_sal,		        \
  void(LHS &, expr::Ternary<OP, B1, B2, B3, true> const &)>             \
{									\
  static char const* name() { return "Expr_SAL_VVV"; }			\
									\
  typedef expr::Ternary<OP, B1, B2, B3, true> RHS;	                \
									\
  typedef typename LHS::value_type lhs_value_type;			\
									\
  typedef typename impl::sal::Effective_value_type<LHS>::type eff_lhs_t;\
  typedef typename impl::sal::Effective_value_type<B1>::type eff_1_t;   \
  typedef typename impl::sal::Effective_value_type<B2>::type eff_2_t;   \
  typedef typename impl::sal::Effective_value_type<B3>::type eff_3_t;   \
									\
  typedef typename impl::Adjust_layout_dim<			    	\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    lhs_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<			     	\
      1, typename impl::Block_layout<B1>::layout_type>::type	        \
    block1_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B2>::layout_type>::type		\
    block2_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B3>::layout_type>::type		\
    block3_lp;								\
  									\
  static bool const ct_valid = 						\
    (!impl::Is_expr_block<B1>::value || impl::Is_scalar_block<B1>::value) &&\
    (!impl::Is_expr_block<B2>::value || impl::Is_scalar_block<B2>::value) &&\
    (!impl::Is_expr_block<B3>::value || impl::Is_scalar_block<B3>::value) &&\
     impl::sal::Is_op3_supported<OP, eff_1_t, eff_2_t, eff_3_t, eff_lhs_t>::value&&\
     /* check that direct access is supported */			\
     impl::Ext_data_cost<LHS>::value == 0 &&				\
     (impl::Ext_data_cost<B1>::value == 0 || impl::Is_scalar_block<B1>::value) &&\
     (impl::Ext_data_cost<B2>::value == 0 || impl::Is_scalar_block<B2>::value) &&\
     (impl::Ext_data_cost<B3>::value == 0 || impl::Is_scalar_block<B3>::value);\
									\
  static bool rt_valid(LHS &, RHS const &)			        \
  {									\
    /* SAL supports all strides */					\
    return true;							\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::sal::Ext_wrapper<LHS, lhs_lp> ext_lhs(lhs, impl::SYNC_OUT);   \
    impl::sal::Ext_wrapper<B1, block1_lp> ext_1(rhs.arg1(), impl::SYNC_IN);\
    impl::sal::Ext_wrapper<B2, block2_lp> ext_2(rhs.arg2(), impl::SYNC_IN);\
    impl::sal::Ext_wrapper<B3, block3_lp> ext_3(rhs.arg3(), impl::SYNC_IN);\
    FUN(typename impl::sal::Ext_wrapper<B1, block1_lp>::sal_type(ext_1),\
        typename impl::sal::Ext_wrapper<B2, block1_lp>::sal_type(ext_2),\
        typename impl::sal::Ext_wrapper<B3, block1_lp>::sal_type(ext_3),\
	typename impl::sal::Ext_wrapper<LHS, lhs_lp>::sal_type(ext_lhs),\
	lhs.size());							\
  }									\
};

VSIP_IMPL_SAL_VVV_EXPR(expr::op::Ma,  impl::sal::vma)
VSIP_IMPL_SAL_VVV_EXPR(expr::op::Msb, impl::sal::vmsb)
VSIP_IMPL_SAL_VVV_EXPR(expr::op::Am,  impl::sal::vam)
VSIP_IMPL_SAL_VVV_EXPR(expr::op::Sbm, impl::sal::vsbm)

/// Ternary expressions, F(V).V.V
#define VSIP_IMPL_SAL_fVVV_EXPR(OP, UOP, FUN, LOOKUP_OP)		\
template <typename LHS, typename B1, typename B2, typename B3>		\
struct Evaluator<op::assign<1>, be::mercury_sal,			\
  void(LHS &,     					                \
  expr::Ternary<OP, expr::Unary<UOP, B1, true> const, B2, B3, true> const &)>\
{									\
  static char const* name() { return "Expr_SAL_fVVV"; }			\
									\
  typedef expr::Ternary<OP, expr::Unary<UOP, B1, true> const, B2, B3, true> RHS; \
  typedef typename LHS::value_type lhs_value_type;			\
									\
  typedef typename impl::sal::Effective_value_type<LHS>::type eff_lhs_t;\
  typedef typename impl::sal::Effective_value_type<B1>::type eff_1_t;   \
  typedef typename impl::sal::Effective_value_type<B2>::type eff_2_t;   \
  typedef typename impl::sal::Effective_value_type<B3>::type eff_3_t;   \
									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    lhs_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<	       			\
      1, typename impl::Block_layout<B1>::layout_type>::type		\
    block1_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B2>::layout_type>::type		\
    block2_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B3>::layout_type>::type		\
    block3_lp;								\
  									\
  static bool const ct_valid = 						\
    (!impl::Is_expr_block<B1>::value || impl::Is_scalar_block<B1>::value) &&\
    (!impl::Is_expr_block<B2>::value || impl::Is_scalar_block<B2>::value) &&\
    (!impl::Is_expr_block<B3>::value || impl::Is_scalar_block<B3>::value) &&\
     impl::sal::Is_op3_supported<LOOKUP_OP, eff_1_t, eff_2_t, eff_3_t,	\
                           eff_lhs_t>::value&&				\
     /* check that direct access is supported */			\
     impl::Ext_data_cost<LHS>::value == 0 &&				\
     (impl::Ext_data_cost<B1>::value == 0 || impl::Is_scalar_block<B1>::value) &&\
     (impl::Ext_data_cost<B2>::value == 0 || impl::Is_scalar_block<B2>::value) &&\
     (impl::Ext_data_cost<B3>::value == 0 || impl::Is_scalar_block<B3>::value);\
									\
  static bool rt_valid(LHS &, RHS const&)			        \
  {									\
    /* SAL supports all strides */					\
    return true;							\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {	                                                                \
    using namespace impl;						\
    sal::Ext_wrapper<LHS, lhs_lp>  ext_lhs(lhs, SYNC_OUT);              \
    sal::Ext_wrapper<B1, block1_lp> ext_1(rhs.arg1().arg(), SYNC_IN);   \
    sal::Ext_wrapper<B2, block2_lp> ext_2(rhs.arg2(), SYNC_IN);         \
    sal::Ext_wrapper<B3, block3_lp> ext_3(rhs.arg3(), SYNC_IN);         \
    FUN(typename sal::Ext_wrapper<B1, block1_lp>::sal_type(ext_1),	\
        typename sal::Ext_wrapper<B2, block2_lp>::sal_type(ext_2),	\
        typename sal::Ext_wrapper<B3, block3_lp>::sal_type(ext_3),	\
	typename sal::Ext_wrapper<LHS, lhs_lp>::sal_type(ext_lhs),	\
	lhs.size());							\
  }									\
};

VSIP_IMPL_SAL_fVVV_EXPR(expr::op::Ma, expr::op::Conj, impl::sal::vcma, impl::sal::cma_token)

// Nested binary expressions, VV_V
#define VSIP_IMPL_SAL_VV_V_EXPR(OP, OP1, OP2, FUN)			\
template <typename LHS, typename B1, typename B2, typename B3> 		\
struct Evaluator<op::assign<1>,                                         \
  typename impl::enable_if_c<impl::Is_leaf_block<B1>::value &&          \
			     impl::Is_leaf_block<B2>::value &&		\
			     impl::Is_leaf_block<B3>::value,		\
			     be::mercury_sal>::type,			\
  void(LHS &, expr::Binary<OP2, expr::Binary<OP1, B1, B2, true> const,	\
                           B3, true> const &)>				\
{									\
  static char const* name() { return "Expr_SAL_VV_V"; }			\
									\
  typedef expr::Binary<OP2,                                             \
                       expr::Binary<OP1, B1, B2, true> const,           \
                       B3, true> RHS;					\
									\
  typedef typename LHS::value_type lhs_value_type;			\
									\
  typedef typename impl::sal::Effective_value_type<LHS>::type eff_lhs_t;\
  typedef typename impl::sal::Effective_value_type<B1>::type eff_1_t;   \
  typedef typename impl::sal::Effective_value_type<B2>::type eff_2_t;   \
  typedef typename impl::sal::Effective_value_type<B3>::type eff_3_t;   \
									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    lhs_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B1>::layout_type>::type		\
    block1_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B2>::layout_type>::type		\
    block2_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B3>::layout_type>::type		\
    block3_lp;								\
  									\
  static bool const ct_valid = 						\
    (!impl::Is_expr_block<B1>::value || impl::Is_scalar_block<B1>::value) &&\
    (!impl::Is_expr_block<B2>::value || impl::Is_scalar_block<B2>::value) &&\
    (!impl::Is_expr_block<B3>::value || impl::Is_scalar_block<B3>::value) &&\
     impl::sal::Is_op3_supported<OP, eff_1_t, eff_2_t, eff_3_t, eff_lhs_t>::value&&\
     /* check that direct access is supported */			\
     impl::Ext_data_cost<LHS>::value == 0 &&				\
     (impl::Ext_data_cost<B1>::value == 0 || impl::Is_scalar_block<B1>::value) &&\
     (impl::Ext_data_cost<B2>::value == 0 || impl::Is_scalar_block<B2>::value) &&\
     (impl::Ext_data_cost<B3>::value == 0 || impl::Is_scalar_block<B3>::value);\
									\
  static bool rt_valid(LHS &, RHS const &)       			\
  {									\
    /* SAL supports all strides */					\
    return true;							\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)		        	\
  {									\
    using namespace impl;                                               \
    sal::Ext_wrapper<LHS, lhs_lp> ext_lhs(lhs, SYNC_OUT);	        \
    sal::Ext_wrapper<B1, block1_lp> ext_1(rhs.arg1().arg1(), SYNC_IN);  \
    sal::Ext_wrapper<B2, block2_lp> ext_2(rhs.arg1().arg2(), SYNC_IN);  \
    sal::Ext_wrapper<B3, block3_lp> ext_3(rhs.arg2(), SYNC_IN);	        \
    FUN(typename sal::Ext_wrapper<B1, block1_lp>::sal_type(ext_1),	\
        typename sal::Ext_wrapper<B2, block2_lp>::sal_type(ext_2),	\
        typename sal::Ext_wrapper<B3, block3_lp>::sal_type(ext_3),	\
	typename sal::Ext_wrapper<LHS, lhs_lp>::sal_type(ext_lhs),	\
	lhs.size());							\
  }									\
};

// Nested binary expressions, V_VV
#define VSIP_IMPL_SAL_V_VV_EXPR(OP, OP1, OP2, FUN)			\
template <typename LHS, typename B1, typename B2, typename B3>		\
struct Evaluator<op::assign<1>,                                         \
  typename impl::enable_if_c<impl::Is_leaf_block<B1>::value &&          \
			     impl::Is_leaf_block<B2>::value &&		\
			     impl::Is_leaf_block<B3>::value,		\
			     be::mercury_sal>::type,			\
  void(LHS &, expr::Binary<OP2, B1, expr::Binary<OP1, B2, B3, true> const,\
                           true> const &)>			      	\
 {									\
  static char const* name() { return "Expr_SAL_V_VV"; }			\
									\
  typedef expr::Binary<OP2, B1, expr::Binary<OP1, B2, B3, true> const, true> RHS;\
									\
  typedef typename LHS::value_type lhs_value_type;			\
									\
  typedef typename impl::sal::Effective_value_type<LHS>::type eff_lhs_t;\
  typedef typename impl::sal::Effective_value_type<B1>::type eff_1_t;   \
  typedef typename impl::sal::Effective_value_type<B2>::type eff_2_t;   \
  typedef typename impl::sal::Effective_value_type<B3>::type eff_3_t;   \
									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    lhs_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B1>::layout_type>::type		\
    block1_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B2>::layout_type>::type		\
    block2_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B3>::layout_type>::type	        \
    block3_lp;								\
  									\
  static bool const ct_valid = 						\
    (!impl::Is_expr_block<B1>::value || impl::Is_scalar_block<B1>::value) &&\
    (!impl::Is_expr_block<B2>::value || impl::Is_scalar_block<B2>::value) &&\
    (!impl::Is_expr_block<B3>::value || impl::Is_scalar_block<B3>::value) &&\
     impl::sal::Is_op3_supported<OP, eff_2_t, eff_3_t, eff_1_t, eff_lhs_t>::value&&\
     /* check that direct access is supported */			\
     impl::Ext_data_cost<LHS>::value == 0 &&				\
     (impl::Ext_data_cost<B1>::value == 0 || impl::Is_scalar_block<B1>::value) &&\
     (impl::Ext_data_cost<B2>::value == 0 || impl::Is_scalar_block<B2>::value) &&\
     (impl::Ext_data_cost<B3>::value == 0 || impl::Is_scalar_block<B3>::value);\
									\
  static bool rt_valid(LHS &, RHS const &)       			\
  {									\
    /* SAL supports all strides */					\
    return true;							\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)		   	        \
  {									\
    using namespace impl;                                               \
    sal::Ext_wrapper<LHS, lhs_lp>  ext_lhs(lhs, SYNC_OUT);              \
    sal::Ext_wrapper<B1, block1_lp> ext_1(rhs.arg1(), SYNC_IN);         \
    sal::Ext_wrapper<B2, block2_lp> ext_2(rhs.arg2().arg1(), SYNC_IN);  \
    sal::Ext_wrapper<B3, block3_lp> ext_3(rhs.arg2().arg2(), SYNC_IN);  \
    FUN(typename sal::Ext_wrapper<B2, block2_lp>::sal_type(ext_2),	\
        typename sal::Ext_wrapper<B3, block3_lp>::sal_type(ext_3),	\
        typename sal::Ext_wrapper<B1, block1_lp>::sal_type(ext_1),	\
	typename sal::Ext_wrapper<LHS, lhs_lp>::sal_type(ext_lhs),	\
	lhs.size());							\
  }									\
};

VSIP_IMPL_SAL_VV_V_EXPR(expr::op::Ma,  expr::op::Mult, expr::op::Add,  impl::sal::vma)
VSIP_IMPL_SAL_VV_V_EXPR(expr::op::Msb, expr::op::Mult, expr::op::Sub,  impl::sal::vmsb)
VSIP_IMPL_SAL_VV_V_EXPR(expr::op::Am,  expr::op::Add,  expr::op::Mult, impl::sal::vam)
VSIP_IMPL_SAL_VV_V_EXPR(expr::op::Sbm, expr::op::Sub,  expr::op::Mult, impl::sal::vsbm)

// V OP2 (V OP1 V)
VSIP_IMPL_SAL_V_VV_EXPR(expr::op::Ma,  expr::op::Mult, expr::op::Add,  impl::sal::vma)
VSIP_IMPL_SAL_V_VV_EXPR(expr::op::Am,  expr::op::Add,  expr::op::Mult, impl::sal::vam)
VSIP_IMPL_SAL_V_VV_EXPR(expr::op::Sbm, expr::op::Sub,  expr::op::Mult, impl::sal::vsbm)

// Nested binary expressions, f(V)V_V
#define VSIP_IMPL_SAL_fVV_V_EXPR(OP, OP1, OP2, UOP, FUN)		\
template <typename LHS, typename B1, typename B2, typename B3>		\
struct Evaluator<op::assign<1>, be::mercury_sal,			\
  void(LHS &,                                                           \
       expr::Binary<OP2, expr::Binary<OP1,                              \
                                      expr::Unary<UOP, B1, true> const, \
   	   	   	     	      B2, true> const,		        \
                    B3, true> const &)>				        \
{									\
  static char const* name() { return "Expr_SAL_fVV_V"; }		\
									\
  typedef expr::Binary<OP2,						\
                       expr::Binary<OP1,				\
		                    expr::Unary<UOP, B1, true> const,   \
                                    B2, true> const,		        \
                       B3, true>					\
	RHS;							        \
									\
  typedef typename LHS::value_type lhs_value_type;			\
									\
  typedef typename impl::sal::Effective_value_type<LHS>::type eff_lhs_t;\
  typedef typename impl::sal::Effective_value_type<B1>::type eff_1_t;   \
  typedef typename impl::sal::Effective_value_type<B2>::type eff_2_t;   \
  typedef typename impl::sal::Effective_value_type<B3>::type eff_3_t;   \
									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    lhs_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B1>::layout_type>::type		\
    block1_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B2>::layout_type>::type		\
    block2_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B3>::layout_type>::type		\
    block3_lp;								\
  									\
  static bool const ct_valid = 						\
    (!impl::Is_expr_block<B1>::value || impl::Is_scalar_block<B1>::value) &&\
    (!impl::Is_expr_block<B2>::value || impl::Is_scalar_block<B2>::value) &&\
    (!impl::Is_expr_block<B3>::value || impl::Is_scalar_block<B3>::value) &&\
     impl::sal::Is_op3_supported<OP, eff_1_t, eff_2_t, eff_3_t, eff_lhs_t>::value&&\
     /* check that direct access is supported */			\
     impl::Ext_data_cost<LHS>::value == 0 &&				\
     (impl::Ext_data_cost<B1>::value == 0 || impl::Is_scalar_block<B1>::value) &&\
     (impl::Ext_data_cost<B2>::value == 0 || impl::Is_scalar_block<B2>::value) &&\
     (impl::Ext_data_cost<B3>::value == 0 || impl::Is_scalar_block<B3>::value);\
									\
  static bool rt_valid(LHS &, RHS const &)			        \
  {									\
    /* SAL supports all strides */					\
    return true;							\
  }									\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    using namespace impl;                                               \
    sal::Ext_wrapper<LHS, lhs_lp>  ext_lhs(lhs, SYNC_OUT);	        \
    sal::Ext_wrapper<B1, block1_lp> ext_1(rhs.arg1().arg1().arg(),SYNC_IN);\
    sal::Ext_wrapper<B2, block2_lp> ext_2(rhs.arg1().arg2(), SYNC_IN);	\
    sal::Ext_wrapper<B3, block3_lp> ext_3(rhs.arg2(), SYNC_IN);         \
    FUN(typename sal::Ext_wrapper<B1, block1_lp>::sal_type(ext_1),	\
        typename sal::Ext_wrapper<B2, block2_lp>::sal_type(ext_2),	\
        typename sal::Ext_wrapper<B3, block3_lp>::sal_type(ext_3),	\
	typename sal::Ext_wrapper<LHS, lhs_lp>::sal_type(ext_lhs),	\
	lhs.size());							\
  }									\
};

VSIP_IMPL_SAL_fVV_V_EXPR(impl::sal::cma_token,
			 expr::op::Mult, expr::op::Add, expr::op::Conj,
			 impl::sal::vcma)

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_SAL_EVAL_ELEMENTWISE_HPP

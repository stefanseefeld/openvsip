/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/expr/assign.hpp
    @author  Stefan Seefeld
    @date    2009-07-13
    @brief   VSIPL++ Library: block expression assignment.
*/

#ifndef VSIP_OPT_EXPR_ASSIGN_HPP
#define VSIP_OPT_EXPR_ASSIGN_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/expr/eval_mcopy.hpp>
#include <vsip/opt/expr/eval_mdim.hpp>
#include <vsip/opt/expr/eval_dense.hpp>
#include <vsip/opt/expr/eval_fastconv.hpp>
#include <vsip/opt/expr/ops_info.hpp>
#include <vsip/core/profile.hpp>

#ifdef VSIP_IMPL_HAVE_IPP
#  include <vsip/opt/ipp/bindings.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_SAL
#  include <vsip/opt/sal/bindings.hpp>
#endif
#ifdef VSIP_IMPL_CBE_SDK
#  include <vsip/opt/cbe/ppu/assign.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_CUDA
#  include <vsip/opt/cuda/assign.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_SIMD_LOOP_FUSION
#  include <vsip/opt/simd/expr_evaluator.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_SIMD_UNALIGNED_LOOP_FUSION
#  include <vsip/opt/simd/eval_unaligned.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_SIMD_GENERIC
#  include <vsip/opt/simd/eval_generic.hpp>
#endif

namespace vsip
{
namespace impl
{
template <typename LHS, typename RHS,
	  dimension_type D = LHS::dim,
	  typename O = typename Block_layout<LHS>::order_type>
struct Loop_fusion_assign;

template <typename LHS, typename RHS, typename O>
struct Loop_fusion_assign<LHS, RHS, 1, O>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size = lhs.size(1, 0);
    for (index_type i=0; i<size; ++i)
      lhs.put(i, rhs.get(i));
  }
};

template <typename LHS, typename RHS>
struct Loop_fusion_assign<LHS, RHS, 2, row2_type>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const rows = lhs.size(2, 0);
    length_type const cols = lhs.size(2, 1);
    for (index_type i=0; i<rows; ++i)
      for (index_type j=0; j<cols; ++j)
	lhs.put(i, j, rhs.get(i, j));
  }
};

template <typename LHS, typename RHS>
struct Loop_fusion_assign<LHS, RHS, 2, col2_type>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const rows = lhs.size(2, 0);
    length_type const cols = lhs.size(2, 1);
    for (index_type j=0; j<cols; ++j)
      for (index_type i=0; i<rows; ++i)
	lhs.put(i, j, rhs.get(i, j));
  }
};

template <typename LHS, typename RHS>
struct Loop_fusion_assign<LHS, RHS, 3, tuple<0,1,2> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type i=0; i<size0; ++i)
      for (index_type j=0; j<size1; ++j)
	for (index_type k=0; k<size2; ++k)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

template <typename LHS, typename RHS>
struct Loop_fusion_assign<LHS, RHS, 3, tuple<0,2,1> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type i=0; i<size0; ++i)
      for (index_type k=0; k<size2; ++k)
	for (index_type j=0; j<size1; ++j)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

template <typename LHS, typename RHS>
struct Loop_fusion_assign<LHS, RHS, 3, tuple<1,0,2> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type j=0; j<size1; ++j)
      for (index_type i=0; i<size0; ++i)
	for (index_type k=0; k<size2; ++k)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

template <typename LHS, typename RHS>
struct Loop_fusion_assign<LHS, RHS, 3, tuple<1,2,0> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type j=0; j<size1; ++j)
      for (index_type k=0; k<size2; ++k)
	for (index_type i=0; i<size0; ++i)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

template <typename LHS, typename RHS>
struct Loop_fusion_assign<LHS, RHS, 3, tuple<2,0,1> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type k=0; k<size2; ++k)
      for (index_type i=0; i<size0; ++i)
	for (index_type j=0; j<size1; ++j)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

template <typename LHS, typename RHS>
struct Loop_fusion_assign<LHS, RHS, 3, tuple<2,1,0> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type k=0; k<size2; ++k)
      for (index_type j=0; j<size1; ++j)
	for (index_type i=0; i<size0; ++i)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
/// Map assignment to profile::dispatch
template <dimension_type D>
struct Profile_feature<op::assign<D> >
{
  static unsigned int const value = impl::profile::dispatch;
};

/// Specialize assignment profiling, as the op-counting for that is
/// handled separately.
template <dimension_type D, typename LHS, typename RHS, typename B>
struct Profile_policy<op::assign<D>, void(LHS &, RHS const &), B>
{
  typedef impl::profile::Scope<impl::profile::dispatch> scope_type;
  typedef Evaluator<op::assign<D>, B, void(LHS &, RHS const &)> evaluator_type;
  Profile_policy(LHS &, RHS const &rhs)
    : scope(expr::op_name(evaluator_type::name(), rhs),
	    expr::Ops_per_point<RHS>::value == 0 ?
	    // If ops_per_point is 0, then assume that operations
	    // is a copy and record the number of bytes written.
	    sizeof(typename LHS::value_type) * expr::Ops_per_point<RHS>::size(rhs) :
	    // Otherwise, record the number of flops.
	    expr::Ops_per_point<RHS>::value * expr::Ops_per_point<RHS>::size(rhs))
  {}
    
  scope_type scope;  
};

/// Copy data using memcpy
template <typename LHS, typename RHS>
struct Evaluator<op::assign<1>, be::copy, void(LHS &, RHS const &)>
{
  static char const *name() { return "Expr_Copy";}

  typedef typename 
  impl::Adjust_layout_dim<1, typename impl::Block_layout<LHS>::layout_type>::type
  lhs_layout_type;

  typedef typename 
  impl::Adjust_layout_dim<1, typename impl::Block_layout<RHS>::layout_type>::type
  rhs_layout_type;

  static bool const ct_valid = 
    LHS::dim == 1 && RHS::dim == 1 &&
    impl::Ext_data_cost<LHS>::value == 0 &&
    impl::Ext_data_cost<RHS>::value == 0 &&
    !impl::Is_split_block<LHS>::value &&
    !impl::Is_split_block<RHS>::value;

  static bool rt_valid(LHS &, RHS const &) { return true;}
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    impl::Ext_data<LHS, lhs_layout_type> ext_lhs(lhs, impl::SYNC_OUT);
    impl::Ext_data<RHS, rhs_layout_type> ext_rhs(rhs, impl::SYNC_IN);

    typename impl::Ext_data<LHS, lhs_layout_type>::raw_ptr_type ptr1 = ext_lhs.data();
    typename impl::Ext_data<RHS, rhs_layout_type>::raw_ptr_type ptr2 = ext_rhs.data();

    stride_type stride1 = ext_lhs.stride(0);
    stride_type stride2 = ext_rhs.stride(0);
    length_type size    = ext_lhs.size(0);
    assert(size <= ext_rhs.size(0));

    if (impl::Type_equal<typename LHS::value_type,
	                 typename RHS::value_type>::value &&
	stride1 == 1 && stride2 == 1)
    {
      memcpy(ptr1, ptr2, size*sizeof(typename LHS::value_type));
    }
    else
    {
      while (size--)
      {
	*ptr1 = *ptr2;
	ptr1 += stride1;
	ptr2 += stride2;
      }
    }
  }
};

/// Simple loop-fusion assignment.
template <dimension_type D, typename LHS, typename RHS>
struct Evaluator<op::assign<D>, be::loop_fusion, void(LHS &, RHS const &)>
{
  static char const *name() { return "Expr_Loop"; }
  static bool const ct_valid = true;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  
  static void exec(LHS &lhs, RHS const &rhs)
  { vsip::impl::Loop_fusion_assign<LHS, RHS, D>::exec(lhs, rhs);}
};

/// Specialization for (non-elementwise) Unary expressions, using the RBO evaluation.
template <dimension_type D, typename LHS,
	  template <typename> class Operation,
	  typename Block>
struct Evaluator<op::assign<D>, be::rbo_expr,
		 void(LHS &, expr::Unary<Operation, Block> const &)>
{
  typedef expr::Unary<Operation, Block> RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static char const *name() { return "op::rbo_expr";}

  static bool const ct_valid = true;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs) { rhs.apply(lhs);}
};

/// Specialization for (non-elementwise) Binary expressions, using the RBO evaluation.
template <dimension_type D, typename LHS,
	  template <typename, typename> class Operation,
	  typename Block1, typename Block2>
struct Evaluator<op::assign<D>, be::rbo_expr,
		 void(LHS &, expr::Binary<Operation, Block1, Block2> const &)>
{
  typedef expr::Binary<Operation, Block1, Block2> RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static char const *name() { return "op::rbo_expr";}

  static bool const ct_valid = true;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs) { rhs.apply(lhs);}
};

/// Specialization for (non-elementwise) Ternary expressions, using the RBO evaluation.
template <dimension_type D, typename LHS,
	  template <typename, typename, typename> class Operation,
	  typename Block1, typename Block2, typename Block3>
struct Evaluator<op::assign<D>, be::rbo_expr,
		 void(LHS &, expr::Ternary<Operation, Block1, Block2, Block3> const &)>
{
  typedef expr::Ternary<Operation, Block1, Block2, Block3> RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static char const *name() { return "op::rbo_expr";}

  static bool const ct_valid = true;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs) { rhs.apply(lhs);}
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl


#endif

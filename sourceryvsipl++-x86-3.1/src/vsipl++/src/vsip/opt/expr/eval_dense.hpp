/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/expr/eval_dense.hpp
    @author  Jules Bergmann
    @date    2006-06-05
    @brief   VSIPL++ Library: Evaluate a dense multi-dimensional expression
                              as a vector expression.
*/

#ifndef VSIP_OPT_EXPR_EVAL_DENSE_HPP
#define VSIP_OPT_EXPR_EVAL_DENSE_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/core/storage.hpp>
#include <vsip/dda.hpp>
#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/expr/redim_block.hpp>

namespace vsip
{
namespace impl
{

/// Expression reduction to determine if an expression consists of
/// dense data at the leaves (either blocks with stride_unit_dense 
/// packing, subviews that are dense, or scalar_blocks).  Check is
/// done at runtime, checking for gaps in highest-dimension stride.
struct Reduce_is_expr_dense
{
  template <dimension_type            Dim0,
	    typename                  T>
  bool
  apply(expr::Scalar<Dim0, T> const&)
  {
    return true;
  }

  template <template <typename> class O, typename B>
  bool
  apply(expr::Unary<O, B, true> const &b) { return apply(b.arg());}

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  bool
  apply(expr::Binary<Operation, LBlock, RBlock, true> const& blk)
  {
    return apply(blk.arg1()) && apply(blk.arg2());
  }

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  bool
  apply(expr::Ternary<Operation, Block1, Block2, Block3, true> const& blk)
  {
    return apply(blk.arg1()) && apply(blk.arg2()) && apply(blk.arg3());
  }

  // Leaf combine function.
  template <typename BlockT>
  bool
  apply(BlockT const&, false_type) const
  {
    return false;
  }

  // Leaf combine function.
  template <typename BlockT>
  bool
  apply(BlockT const& block, true_type) const
  {
    typedef typename get_block_layout<BlockT>::order_type order_type;

    dda::Data<BlockT, dda::in> ext(block);

    if (get_block_layout<BlockT>::dim == 1)
      return ext.stride(0) == 1;
    else if (get_block_layout<BlockT>::dim == 2)
      return (ext.stride(order_type::impl_dim0) ==
	      static_cast<stride_type>(ext.size(order_type::impl_dim1)))
	     && ext.stride(order_type::impl_dim1) == 1;
    else if (get_block_layout<BlockT>::dim == 3)
      return (ext.stride(order_type::impl_dim0) ==
	      static_cast<stride_type>(ext.size(order_type::impl_dim1) *
				       ext.size(order_type::impl_dim2)))
             && ext.stride(order_type::impl_dim2) == 1;
    else return false;
  }

  // Leaf combine function.
  template <typename BlockT>
  bool
  apply(BlockT const& block) const
  {
    return apply(block, integral_constant<bool, dda::Data<BlockT, dda::in>::ct_cost == 0>());
  }
};

/// Helper function to apply Reduce_is_expr_dense reduction.
template <typename BlockT>
bool
is_expr_dense(BlockT& blk)
{
  Reduce_is_expr_dense obj;
  return obj.apply(blk);
}




/// Reduction to check if all leaf blocks have dimension-order
/// equivalent to OrderT.
template <typename OrderT>
struct Reduce_is_same_dim_order
{
public:
  template <typename BlockT>
  struct leaf_node
  {
    typedef integral_constant<bool, is_same<typename get_block_layout<BlockT>::order_type,
					    OrderT>::value> type;
  };

  template <dimension_type Dim0,
	    typename       T>
  struct leaf_node<expr::Scalar<Dim0, T> >
  {
    typedef true_type type;
  };

  template <dimension_type Dim0,
	    typename       T>
  struct leaf_node<expr::Scalar<Dim0, T> const>
  {
    typedef true_type type;
  };

  template <template <typename> class Operation,
	    typename Block>
  struct unary_node
  {
    typedef Block type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct binary_node
  {
    typedef integral_constant<bool, LBlock::value && RBlock::value> type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct ternary_node
  {
    typedef integral_constant<bool, Block1::value && Block2::value && Block3::value> type;
  };

  template <typename Block>
  struct transform
  {
    typedef typename leaf_node<Block>::type type;
  };

  template <template <typename> class O, typename B>
  struct transform<expr::Unary<O, B, true> const>
  {
    typedef typename unary_node<O, typename transform<B>::type>::type
    type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct transform<expr::Binary<Operation, LBlock, RBlock, true> const>
  {
    typedef typename binary_node<Operation,
				 typename transform<LBlock>::type,
				 typename transform<RBlock>::type>
    ::type type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>
  {
    typedef typename ternary_node<Operation,
				  typename transform<Block1>::type,
				  typename transform<Block2>::type,
				  typename transform<Block3>::type>
    ::type type;
  };
};


template <typename OrderT,
	  typename BlockT>
struct Is_same_dim_order
{
  static bool const value =
    Reduce_is_same_dim_order<OrderT>::template transform<BlockT>::type::value;
};



/// Reduction to determine if all leaf blocks of an expression support
/// direct access (cost == 0).
struct Reduce_is_expr_direct_access
{
public:
  template <typename BlockT>
  struct leaf_node
  {
    typedef integral_constant<bool, dda::Data<BlockT, dda::in>::ct_cost == 0> type;
  };

  template <dimension_type Dim0,
	    typename       T>
  struct leaf_node<expr::Scalar<Dim0, T> >
  {
    typedef true_type type;
  };

  template <dimension_type Dim0,
	    typename       T>
  struct leaf_node<expr::Scalar<Dim0, T> const>
  {
    typedef true_type type;
  };

  template <template <typename> class Operation,
	    typename Block>
  struct unary_node
  {
    typedef Block type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct binary_node
  {
    typedef integral_constant<bool, LBlock::value && RBlock::value> type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct ternary_node
  {
    typedef integral_constant<bool, Block1::value && Block2::value && Block3::value> type;
  };

  template <typename Block>
  struct transform
  {
    typedef typename leaf_node<Block>::type type;
  };

  template <template <typename> class O, typename B>
  struct transform<expr::Unary<O, B, true> const>
  {
    typedef typename unary_node<O, typename transform<B>::type>::type
    type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct transform<expr::Binary<Operation, LBlock, RBlock, true> const>
  {
    typedef typename binary_node<Operation,
				 typename transform<LBlock>::type,
				 typename transform<RBlock>::type>
    ::type type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>
  {
    typedef typename ternary_node<Operation,
				  typename transform<Block1>::type,
				  typename transform<Block2>::type,
				  typename transform<Block3>::type>
				::type type;
  };
};

/// Report whether the expression is entirely composed of blocks that
/// provide direct data access.
template <typename BlockT>
struct Is_expr_direct_access
{
  static bool const value =
    Reduce_is_expr_direct_access::template transform<BlockT>::type::value;
};

} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

/// Evaluator to convert dense multi-dimensional expressions into
/// 1 dimensional expressions.
template <dimension_type D, typename LHS, typename RHS>
struct Evaluator<op::assign<D>, be::dense_expr, void(LHS &, RHS const &)>
{
  static char const *name() { return "be::dense_expr";}

  static bool const ct_valid =
    D > 1 &&
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    impl::Is_expr_direct_access<RHS const>::value &&
    impl::Is_same_dim_order<typename get_block_layout<LHS>::order_type,
			    RHS const>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  { return impl::is_expr_dense(lhs) && impl::is_expr_dense(rhs);}

  static void exec(LHS &lhs, RHS const &rhs)
  {
    using impl::Redim_expr;
    typedef typename Redim_expr<1>::template transform<RHS const>::type
      new_rhs_type;
    typedef typename Redim_expr<1>::template transform<LHS>::type
      new_lhs_type;
    Redim_expr<1> redim;

    // Create an lvalue that dispatch() below can write into.
    typename impl::View_block_storage<new_lhs_type>::plain_type
      new_lhs = redim.apply(const_cast<LHS&>(lhs));

    vsip_csl::dispatch<op::assign<1>, void, new_lhs_type &, new_rhs_type const &>
      (new_lhs, redim.apply(const_cast<RHS const&>(rhs)));
  }
};

/// Evaluator that processes 1D dense expressions via DDA.
template <typename LHS, 
	  template <typename> class OP, typename A>
struct Evaluator<op::assign<1>, be::dense_expr,
  void(LHS &, expr::Unary<OP, A, true> const &)>
{
  typedef expr::Unary<OP, A, true> RHS;
  typedef OP<typename A::value_type> op_type;

  static char const *name() { return "be::dense_expr";}

  static bool const ct_valid = (!impl::is_expr_block<LHS>::value &&
				dda::Data<LHS, dda::out>::ct_cost == 0 &&
				!impl::is_expr_block<A>::value &&
				dda::Data<A, dda::in>::ct_cost == 0);

  static bool rt_valid(LHS &, RHS const &) { return true;}

  // Even though LL and AA can be fully infered from LHS and A,
  // we may not refer to those types here as LHS and A may not be valid
  // at all (and ct_valid above evaluate to 'false').
  template <typename LL, typename AA>
  static void 
  apply(op_type const &op,
	LL lhs, stride_type lhs_stride,
	AA a, stride_type a_stride,
	length_type size)
  {
    typedef typename dda::Data<LHS, dda::out>::storage_type lhs_st;
    typedef typename dda::Data<A, dda::in>::storage_type a_st;
    for (index_type i = 0; i != size; ++i)
      lhs_st::put(lhs, i * lhs_stride, op(a_st::get(a, i * a_stride)));
  }
  static void exec(LHS &lhs, RHS const &rhs)
  {
    // Even though this is a 1D assignment, the blocks needn't be 1D !
    // To complicate things further, LHS and RHS may have different dimensionality
    // (e.g., Dense<2> and Redim<...>)
    dimension_type const lhs_minor = LHS::dim - 1;
    dimension_type const a_minor = A::dim - 1;
    dda::Data<LHS, dda::out> ext_lhs(lhs);
    dda::Data<A, dda::in> ext_a(rhs.arg());
    apply(rhs.operation(),
	  ext_lhs.ptr(), ext_lhs.stride(lhs_minor),
	  ext_a.ptr(), ext_a.stride(a_minor),
	  ext_lhs.size());
  }
};

/// Evaluator that processes 1D dense expressions via DDA.
template <typename LHS, 
	  template <typename, typename> class OP, typename A1, typename A2>
struct Evaluator<op::assign<1>, be::dense_expr,
  void(LHS &, expr::Binary<OP, A1, A2, true> const &)>
{
  typedef expr::Binary<OP, A1, A2, true> RHS;
  typedef OP<typename A1::value_type, typename A2::value_type> op_type;

  static char const *name() { return "be::dense_expr";}

  static bool const ct_valid = (!impl::is_expr_block<LHS>::value &&
				dda::Data<LHS, dda::out>::ct_cost == 0 &&
				!impl::is_expr_block<A1>::value &&
				dda::Data<A1, dda::in>::ct_cost == 0 &&
				!impl::is_expr_block<A2>::value &&
				dda::Data<A2, dda::in>::ct_cost == 0);

  static bool rt_valid(LHS &, RHS const &) { return true;}

  // Even though LL et al. can be fully infered from LHS etc.,
  // we may not refer to those types here as they may not be valid
  // at all (and ct_valid above evaluate to 'false').
  template <typename LL, typename AA1, typename AA2>
  static void 
  apply(op_type const &op,
	LL lhs, stride_type lhs_stride,
	AA1 a1, stride_type a1_stride,
	AA2 a2, stride_type a2_stride,
	length_type size)
  {
    typedef typename dda::Data<LHS, dda::out>::storage_type lhs_st;
    typedef typename dda::Data<A1, dda::in>::storage_type a1_st;
    typedef typename dda::Data<A2, dda::in>::storage_type a2_st;
    for (index_type i = 0; i != size; ++i)
      lhs_st::put(lhs, i * lhs_stride,
		  op(a1_st::get(a1, i * a1_stride),
		     a2_st::get(a2, i * a2_stride)));
  }
  static void exec(LHS &lhs, RHS const &rhs)
  {
    // Even though this is a 1D assignment, the blocks needn't be 1D !
    // To complicate things further, LHS and RHS may have different dimensionality
    // (e.g., Dense<2> and Redim<...>)
    dimension_type const lhs_minor = LHS::dim - 1;
    dimension_type const a1_minor = A1::dim - 1;
    dimension_type const a2_minor = A2::dim - 1;
    dda::Data<LHS, dda::out> ext_lhs(lhs);
    dda::Data<A1, dda::in> ext_a1(rhs.arg1());
    dda::Data<A2, dda::in> ext_a2(rhs.arg2());
    apply(rhs.operation(),
	  ext_lhs.ptr(), ext_lhs.stride(lhs_minor),
	  ext_a1.ptr(), ext_a1.stride(a1_minor),
	  ext_a2.ptr(), ext_a2.stride(a2_minor),
	  ext_lhs.size());
  }
};

/// Evaluator that processes 1D dense expressions via DDA.
template <typename LHS, 
	  template <typename, typename, typename> class OP,
	  typename A1, typename A2, typename A3>
struct Evaluator<op::assign<1>, be::dense_expr,
  void(LHS &, expr::Ternary<OP, A1, A2, A3, true> const &)>
{
  typedef expr::Ternary<OP, A1, A2, A3, true> RHS;
  typedef OP<typename A1::value_type,
	     typename A2::value_type,
	     typename A3::value_type> op_type;

  static char const *name() { return "be::dense_expr";}

  static bool const ct_valid = (!impl::is_expr_block<LHS>::value &&
				dda::Data<LHS, dda::out>::ct_cost == 0 &&
				!impl::is_expr_block<A1>::value &&
				dda::Data<A1, dda::in>::ct_cost == 0 &&
				!impl::is_expr_block<A2>::value &&
				dda::Data<A2, dda::in>::ct_cost == 0 &&
				!impl::is_expr_block<A3>::value &&
				dda::Data<A3, dda::in>::ct_cost == 0);

  static bool rt_valid(LHS &, RHS const &) { return true;}

  // Even though LL et al. can be fully infered from LHS etc.,
  // we may not refer to those types here as they may not be valid
  // at all (and ct_valid above evaluate to 'false').
  template <typename LL, typename AA1, typename AA2, typename AA3>
  static void 
  apply(op_type const &op,
	LL lhs, stride_type lhs_stride,
	AA1 a1, stride_type a1_stride,
	AA2 a2, stride_type a2_stride,
	AA3 a3, stride_type a3_stride,
	length_type size)
  {
    typedef typename dda::Data<LHS, dda::out>::storage_type lhs_st;
    typedef typename dda::Data<A1, dda::in>::storage_type a1_st;
    typedef typename dda::Data<A2, dda::in>::storage_type a2_st;
    typedef typename dda::Data<A3, dda::in>::storage_type a3_st;
    for (index_type i = 0; i != size; ++i)
      lhs_st::put(lhs, i * lhs_stride,
		  op(a1_st::get(a1, i * a1_stride),
		     a2_st::get(a2, i * a2_stride),
		     a3_st::get(a3, i * a3_stride)));
  }
  static void exec(LHS &lhs, RHS const &rhs)
  {
    // Even though this is a 1D assignment, the blocks needn't be 1D !
    // To complicate things further, LHS and RHS may have different dimensionality
    // (e.g., Dense<2> and Redim<...>)
    dimension_type const lhs_minor = LHS::dim - 1;
    dimension_type const a1_minor = A1::dim - 1;
    dimension_type const a2_minor = A2::dim - 1;
    dimension_type const a3_minor = A3::dim - 1;
    dda::Data<LHS, dda::out> ext_lhs(lhs);
    dda::Data<A1, dda::in> ext_a1(rhs.arg1());
    dda::Data<A2, dda::in> ext_a2(rhs.arg2());
    dda::Data<A3, dda::in> ext_a3(rhs.arg3());
    apply(rhs.operation(),
	  ext_lhs.ptr(), ext_lhs.stride(lhs_minor),
	  ext_a1.ptr(), ext_a1.stride(a1_minor),
	  ext_a2.ptr(), ext_a2.stride(a2_minor),
	  ext_a3.ptr(), ext_a3.stride(a3_minor),
	  ext_lhs.size());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

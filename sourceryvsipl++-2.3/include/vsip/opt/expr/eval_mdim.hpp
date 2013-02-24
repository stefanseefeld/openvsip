/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/expr/eval_mdim.hpp
    @author  Jules Bergmann
    @date    2007-08-13
    @brief   VSIPL++ Library: Evaluate a multi-dimensional expression
                              as multiply vector expression.
*/

#ifndef VSIP_OPT_EXPR_EVAL_MDIM_HPP
#define VSIP_OPT_EXPR_EVAL_MDIM_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>

namespace vsip
{
namespace impl
{

/// Expression reduction to determine if an expression contains
/// difficult blocks to handle, such as Rbo blocks.
struct Reduce_is_expr_difficult
{
public:
  template <typename BlockT>
  struct leaf_node
  {
    typedef Bool_type<false> type;
  };

  template <typename BlockT>
  struct transform
  {
    typedef typename leaf_node<BlockT>::type type;
  };

  template <template <typename> class O, typename B>
  struct transform<expr::Unary<O, B, true> const>
  {
    typedef typename transform<B>::type type;
  };

  template <template <typename> class O, typename B>
  struct transform<expr::Unary<O, B, false> const>
  {
    typedef Bool_type<true> type;
  };

  template <dimension_type                Dim0,
	    typename                      LBlock,
	    typename                      RBlock>
  struct transform<expr::Vmmul<Dim0, LBlock, RBlock> const>
  {
//    typedef Bool_type<true> type;
    typedef Bool_type<transform<LBlock>::type::value ||
                      transform<RBlock>::type::value> type;
  };

  template <template <typename, typename> class Operation,
	    typename                      LBlock,
	    typename                      RBlock>
  struct transform<expr::Binary<Operation, LBlock, RBlock, true> const>
  {
    typedef Bool_type<transform<LBlock>::type::value ||
                      transform<RBlock>::type::value> type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename                      Block1,
	    typename                      Block2,
	    typename                      Block3>
  struct transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>
  {
    typedef Bool_type<transform<Block1>::type::value ||
                      transform<Block2>::type::value ||
                      transform<Block2>::type::value> type;
  };
};



// Reduction to extract a 1-dimensional subview from an expression
// with n-dimensional (where n > 1), taking care to push subview
// as "deep" as possible so that library evaluators can still be
// applied.

template <dimension_type FixedDim>
class Subdim_expr
{
public:
  template <typename BlockT>
  struct leaf_node
  {
    typedef Sliced_block<BlockT, FixedDim> type;
  };

  template <dimension_type Dim,
	    typename       T>
  struct leaf_node<expr::Scalar<Dim, T> >
  {
    typedef expr::Scalar<1, T> type;
  };

  template <template <typename> class O, typename B>
  struct unary_node
  {
    typedef expr::Unary<O, B, true> const type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct binary_node
  {
    typedef expr::Binary<Operation, LBlock, RBlock, true> const type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct ternary_node
  {
    typedef expr::Ternary<Operation, Block1, Block2, Block3, true> const type;
  };

  template <typename BlockT>
  struct transform
  {
    typedef typename leaf_node<BlockT>::type type;
  };

  template <template <typename> class O, typename B>
  struct transform<expr::Unary<O, B, true> const>
  {
    typedef typename unary_node<O, typename transform<B>::type>::type type;
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


  template <template <typename> class O, typename B>
  typename transform<expr::Unary<O, B, true> const>::type
  apply(expr::Unary<O, B, true> const &b)
  {
    typedef typename transform<expr::Unary<O, B, true> const>::type
      block_type;
    return block_type(b.operation(), apply(const_cast<B&>(b.arg())));
  }

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  typename transform<expr::Binary<Operation, LBlock, RBlock, true> const>::type
  apply(expr::Binary<Operation, LBlock, RBlock, true> const& blk)
  {
    typedef typename
      transform<expr::Binary<Operation, LBlock, RBlock, true> const>::type
        block_type;
    return block_type(apply(const_cast<LBlock&>(blk.arg1())),
		      apply(const_cast<RBlock&>(blk.arg2())));
  }

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  typename transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>::type
  apply(expr::Ternary<Operation, Block1, Block2, Block3, true> const& blk)
  {
    typedef typename
      transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>::type
        block_type;
    return block_type(apply(const_cast<Block1&>(blk.arg1())),
		      apply(const_cast<Block2&>(blk.arg2())),
		      apply(const_cast<Block3&>(blk.arg3())));
  }

  // Leaf combine function for expr::Scalar.
  template <dimension_type Dim,
	    typename       T>
  typename transform<expr::Scalar<Dim, T> >::type
  apply(expr::Scalar<Dim, T>& block) const
  {
    typedef typename transform<expr::Scalar<Dim, T> >::type type;
    return type(block.value());
  }


  // Leaf combine function.
  template <typename BlockT>
  typename transform<BlockT>::type
  apply(BlockT& block) const
  {
    typedef typename transform<BlockT>::type block_type;
    return block_type(block, idx_);
  }

  // Constructors.
public:
  Subdim_expr(index_type idx) : idx_(idx) {}

  index_type idx_;
};

} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

/// Evaluator to convert multi-dimensional expressions into multiple 1
/// dimensional expressions.  Intended to handle non-dense multi-dim
/// expressions that cannot be handled by Dense_expr_tag.
template <typename LHS, typename RHS>
struct Evaluator<op::assign<2>, be::mdim_expr, void(LHS &, RHS const &)>
{
  static char const *name() { return "Expr_Mdim<2>";}

  static bool const ct_valid =
    !impl::Reduce_is_expr_difficult::template transform<RHS const>::type::value;

  static bool rt_valid(LHS &, RHS const &) { return true;}

  typedef typename impl::Block_layout<LHS>::order_type order_type;
  static dimension_type const fixed_dim = 0;

  typedef typename impl::Proper_type_of<RHS>::type proper_rhs_block_type;

  typedef typename impl::Subdim_expr<fixed_dim>::template
                   transform<proper_rhs_block_type>::type
    new_rhs_type;
  typedef typename impl::Subdim_expr<fixed_dim>::template transform<LHS>::type
    new_lhs_type;

  static new_lhs_type diag_helper_dst(LHS &lhs)
  {
    impl::Subdim_expr<fixed_dim> reduce(0);
    typename impl::View_block_storage<new_lhs_type>::plain_type
      new_lhs = reduce.apply(lhs);
    return new_lhs;
  }

  static new_rhs_type diag_helper_src(RHS const &rhs)
  {
    impl::Subdim_expr<fixed_dim> reduce(0);
    typename impl::View_block_storage<new_rhs_type>::plain_type
      new_rhs = reduce.apply(const_cast<RHS&>(rhs));
    return new_rhs;
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    for (index_type r=0; r<lhs.size(2, fixed_dim); ++r)
    {
      impl::Subdim_expr<fixed_dim> reduce(r);
      // Create an lvalue that dispatch() below can write into.
      typename impl::View_block_storage<new_lhs_type>::plain_type
	new_lhs = reduce.apply(const_cast<LHS&>(lhs));

      vsip_csl::dispatch<op::assign<1>, void, new_lhs_type &, new_rhs_type const &>
	(new_lhs, reduce.apply(const_cast<proper_rhs_block_type&>(rhs)));
    }
  }
};

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_EXPR_EVAL_MDIM_HPP

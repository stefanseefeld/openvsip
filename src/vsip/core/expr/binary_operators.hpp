//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

/// Description
///   Binary operators for VSIPL++ views.

#ifndef VSIP_CORE_EXPR_BINARY_OPERATORS_HPP
#define VSIP_CORE_EXPR_BINARY_OPERATORS_HPP

#include <vsip/core/refcount.hpp>
#include <vsip/core/expr/operations.hpp>
#include <vsip/core/expr/binary_block.hpp>
#include <vsip/core/expr/ternary_block.hpp>
#include <vsip/core/expr/scalar_block.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip/core/promote.hpp>
#include <vsip/core/c++0x.hpp>

namespace vsip_csl
{
namespace expr
{
namespace op
{
// Declare Multiply-Add operation as operator* needs it.
template <typename T1, typename T2, typename T3>
struct Ma;

} // namespace vsip_csl::expr::op
} // namespace vsip_csl::expr
} // namespace vsip_csl

namespace vsip
{
namespace impl
{

/// View_promotion allows the compiler to instantiate
/// binary operators for views that differ in their
/// constness only.
/// A 'type' member is only provided in the special case
/// where the two view template parameters are identical,
/// and thus template parameter substitution will fail
/// in general. 
/// This ('SFINAE') principle makes the compiler consider
/// the binary operator template only for matching view
/// types.
template <typename LView, typename RView>
struct View_promotion
{
  // No 'type' member here as LView and RView are incompatible in general.
};

/// Specialization for View_promotion with two identical
/// View class templates.
template <typename View>
struct View_promotion<View, View>
{
  typedef View type;
};

/// Binary_operator_traits encapsulates the logic to generate an 
/// expression block view for binary operators.
template <template <typename, typename> class Operator,
	  template <typename, typename> class LView, typename LBlock,
 	  template <typename, typename> class RView, typename RBlock,
	  typename Enable = void>
struct Binary_operator_traits
{
  typedef Operator<typename LBlock::value_type, typename RBlock::value_type> 
    operator_type;
  typedef const expr::Binary<Operator, LBlock, RBlock, true> block_type;
  typedef typename block_type::value_type value_type;

  typedef typename 
  View_promotion<typename ViewConversion<LView,
					 value_type,
					 block_type>::const_view_type,
		 typename ViewConversion<RView,
					 value_type,
					 block_type>::const_view_type>::type 
    type; // 'type' is the expected name lazy_enable_if relies on.

  static type create(LBlock const &lblock, RBlock const &rblock)
  {
    block_type block(lblock, rblock);
    return type(block);
  }
};

/// Fuse A * B + C into a Ternary expression block.
template <template <typename, typename> class LView,
	  typename ABlock, typename BBlock,
	  template <typename, typename> class RView,
	  typename CBlock>
struct Binary_operator_traits<expr::op::Add,
  LView, expr::Binary<expr::op::Mult, ABlock, BBlock, true> const,
  RView, CBlock>
{
  typedef expr::Binary<expr::op::Mult, ABlock, BBlock, true> l_block_type;
  typedef expr::Ternary<expr::op::Ma, ABlock, BBlock, CBlock, true> const block_type;
  typedef typename block_type::value_type value_type;

  typedef typename 
  View_promotion<typename ViewConversion<LView,
					 value_type,
					 block_type>::const_view_type,
		 typename ViewConversion<RView,
					 value_type,
					 block_type>::const_view_type>::type 
    type; // 'type' is the expected name lazy_enable_if relies on.

  static type create(l_block_type const &lblock, CBlock const &cblock)
  {
    block_type block(lblock.arg1(), lblock.arg2(), cblock);
    return type(block);
  }
};

/// Fuse A + B * C into a Ternary expression block.
template <template <typename, typename> class LView,
	  typename ABlock,
	  template <typename, typename> class RView,
	  typename BBlock, typename CBlock>
struct Binary_operator_traits<expr::op::Add,
  LView, ABlock,
  RView, expr::Binary<expr::op::Mult, BBlock, CBlock, true> const>
{
  typedef expr::Binary<expr::op::Mult, BBlock, CBlock, true> r_block_type;
  typedef expr::Ternary<expr::op::Ma, BBlock, CBlock, ABlock, true> const block_type;
  typedef typename block_type::value_type value_type;

  typedef typename 
  View_promotion<typename ViewConversion<LView,
					 value_type,
					 block_type>::const_view_type,
		 typename ViewConversion<RView,
					 value_type,
					 block_type>::const_view_type>::type 
    type; // 'type' is the expected name lazy_enable_if relies on.

  static type create(ABlock const &ablock, r_block_type const &rblock)
  {
    block_type block(rblock.arg1(), rblock.arg2(), ablock);
    return type(block);
  }
};

/// Fuse A * B + C * D into a Ternary expression block.
/// (This maps to Ternary<A, B, Binary<B, C> >.)
template <template <typename, typename> class LView,
	  typename ABlock, typename BBlock,
	  template <typename, typename> class RView,
	  typename CBlock, typename DBlock>
struct Binary_operator_traits<expr::op::Add,
  LView, expr::Binary<expr::op::Mult, ABlock, BBlock, true> const,
  RView, expr::Binary<expr::op::Mult, CBlock, DBlock, true> const>
{
  typedef expr::Binary<expr::op::Mult, ABlock, BBlock, true> l_block_type;
  typedef expr::Binary<expr::op::Mult, CBlock, DBlock, true> r_block_type;
  typedef expr::Ternary<expr::op::Ma, ABlock, BBlock, r_block_type const, true> const block_type;
  typedef typename block_type::value_type value_type;

  typedef typename 
  View_promotion<typename ViewConversion<LView,
					 value_type,
					 block_type>::const_view_type,
		 typename ViewConversion<RView,
					 value_type,
					 block_type>::const_view_type>::type 
    type; // 'type' is the expected name lazy_enable_if relies on.

  static type create(l_block_type const &lblock, r_block_type const &rblock)
  {
    block_type block(lblock.arg1(), lblock.arg2(), rblock);
    return type(block);
  }
};
template <template <typename, typename> class View,
 	  typename Type,
	  typename Block,
	  typename S>
inline
typename lazy_enable_if<Is_view_type<View<Type, Block> >,
  Binary_operator_traits<expr::op::Add,
    View, Block,
    View, expr::Scalar<View<Type, Block>::dim, S> const> >::type
operator+(View<Type, Block> const& lhs, S rhs)
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef Binary_operator_traits<expr::op::Add, View, Block, View, scalar_block_type> traits;
  return traits::create(lhs.block(), scalar_block_type(rhs));
}

template <template <typename, typename> class View,
 	  typename Type,
	  typename Block,
	  typename S>
inline
typename lazy_enable_if<Is_view_type<View<Type, Block> >,
  Binary_operator_traits<expr::op::Add,
    View, expr::Scalar<View<Type, Block>::dim, S> const,
    View, Block> >::type
operator+(S lhs, View<Type, Block> const& rhs)
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef Binary_operator_traits<expr::op::Add, View, scalar_block_type, View, Block> traits;
  return traits::create(scalar_block_type(lhs), rhs.block());
}

template <template <typename, typename> class LView,
 	  typename LType,
	  typename LBlock,
 	  template <typename, typename> class RView,
 	  typename RType,
	  typename RBlock>
inline
typename lazy_enable_if_c<Is_view_type<LView<LType, LBlock> >::value &&
			  Is_view_type<RView<RType, RBlock> >::value,
  Binary_operator_traits<expr::op::Add, LView, LBlock, RView, RBlock> >::type
operator+(LView<LType, LBlock> const& lhs, RView<RType, RBlock> const& rhs)
{
  typedef Binary_operator_traits<expr::op::Add, LView, LBlock, RView, RBlock> traits;
  return traits::create(lhs.block(), rhs.block());
}

template <template <typename, typename> class View,
 	  typename Type,
	  typename Block,
	  typename S>
inline
typename lazy_enable_if<Is_view_type<View<Type, Block> >,
  Binary_operator_traits<expr::op::Sub,
    View, Block,
    View, expr::Scalar<View<Type, Block>::dim, S> const> >::type
operator-(View<Type, Block> const& lhs, S rhs)
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef Binary_operator_traits<expr::op::Sub, View, Block, View, scalar_block_type> traits;
  return traits::create(lhs.block(), scalar_block_type(rhs));
}

template <template <typename, typename> class View,
 	  typename Type,
	  typename Block,
	  typename S>
inline
typename lazy_enable_if<Is_view_type<View<Type, Block> >,
  Binary_operator_traits<expr::op::Sub,
    View, expr::Scalar<View<Type, Block>::dim, S> const,
    View, Block> >::type
operator-(S lhs, View<Type, Block> const& rhs)
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef Binary_operator_traits<expr::op::Sub, View, scalar_block_type, View, Block> traits;
  return traits::create(scalar_block_type(lhs), rhs.block());
}

template <template <typename, typename> class LView,
 	  typename LType,
	  typename LBlock,
 	  template <typename, typename> class RView,
 	  typename RType,
	  typename RBlock>
inline
typename lazy_enable_if_c<Is_view_type<LView<LType, LBlock> >::value &&
			  Is_view_type<RView<RType, RBlock> >::value,
  Binary_operator_traits<expr::op::Sub, LView, LBlock, RView, RBlock> >::type
operator-(LView<LType, LBlock> const& lhs, RView<RType, RBlock> const& rhs)
{
  typedef Binary_operator_traits<expr::op::Sub, LView, LBlock, RView, RBlock> traits;
  return traits::create(lhs.block(), rhs.block());
}

template <template <typename, typename> class View,
 	  typename Type,
	  typename Block,
	  typename S>
inline
typename lazy_enable_if<Is_view_type<View<Type, Block> >,
  Binary_operator_traits<expr::op::Mult,
    View, Block,
    View, expr::Scalar<View<Type, Block>::dim, S> const> >::type
operator*(View<Type, Block> const& lhs, S rhs)
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef Binary_operator_traits<expr::op::Mult, View, Block, View, scalar_block_type> traits;
  return traits::create(lhs.block(), scalar_block_type(rhs));
}

template <template <typename, typename> class View,
 	  typename Type,
	  typename Block,
	  typename S>
inline
typename lazy_enable_if<Is_view_type<View<Type, Block> >,
  Binary_operator_traits<expr::op::Mult,
    View, expr::Scalar<View<Type, Block>::dim, S> const,
    View, Block> >::type
operator*(S lhs, View<Type, Block> const& rhs)
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef Binary_operator_traits<expr::op::Mult, View, scalar_block_type, View, Block> traits;
  return traits::create(scalar_block_type(lhs), rhs.block());
}

template <template <typename, typename> class LView,
 	  typename LType,
	  typename LBlock,
 	  template <typename, typename> class RView,
 	  typename RType,
	  typename RBlock>
inline
typename lazy_enable_if_c<Is_view_type<LView<LType, LBlock> >::value &&
			  Is_view_type<RView<RType, RBlock> >::value,
  Binary_operator_traits<expr::op::Mult, LView, LBlock, RView, RBlock> >::type
operator*(LView<LType, LBlock> const& lhs, RView<RType, RBlock> const& rhs)
{
  typedef Binary_operator_traits<expr::op::Mult, LView, LBlock, RView, RBlock> traits;
  return traits::create(lhs.block(), rhs.block());
}

template <template <typename, typename> class View,
 	  typename Type,
	  typename Block,
	  typename S>
inline
typename lazy_enable_if<Is_view_type<View<Type, Block> >,
  Binary_operator_traits<expr::op::Div,
    View, Block,
    View, expr::Scalar<View<Type, Block>::dim, S> const> >::type
operator/(View<Type, Block> const& lhs, S rhs)
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef Binary_operator_traits<expr::op::Div, View, Block, View, scalar_block_type> traits;
  return traits::create(lhs.block(), scalar_block_type(rhs));
}

template <template <typename, typename> class View,
 	  typename Type,
	  typename Block,
	  typename S>
inline
typename lazy_enable_if<Is_view_type<View<Type, Block> >,
  Binary_operator_traits<expr::op::Div,
    View, expr::Scalar<View<Type, Block>::dim, S> const,
    View, Block> >::type
operator/(S lhs, View<Type, Block> const& rhs)
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef Binary_operator_traits<expr::op::Div, View, scalar_block_type, View, Block> traits;
  return traits::create(scalar_block_type(lhs), rhs.block());
}

template <template <typename, typename> class LView,
 	  typename LType,
	  typename LBlock,
 	  template <typename, typename> class RView,
 	  typename RType,
	  typename RBlock>
inline
typename lazy_enable_if_c<Is_view_type<LView<LType, LBlock> >::value &&
			  Is_view_type<RView<RType, RBlock> >::value,
  Binary_operator_traits<expr::op::Div, LView, LBlock, RView, RBlock> >::type
operator/(LView<LType, LBlock> const& lhs, RView<RType, RBlock> const& rhs)
{
  typedef Binary_operator_traits<expr::op::Div, LView, LBlock, RView, RBlock> traits;
  return traits::create(lhs.block(), rhs.block());
}

} // namespace vsip::impl
} // namespace vsip

#endif

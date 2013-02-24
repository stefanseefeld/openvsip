/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/expr/unary_operators.hpp
    @author  Stefan Seefeld
    @date    2005-03-24
    @brief   VSIPL++ Library: Operators to be used with expression templates.

    This file declares Operators to be used with expression templates.
*/

#ifndef VSIP_CORE_EXPR_UNARY_OPERATORS_HPP
#define VSIP_CORE_EXPR_UNARY_OPERATORS_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/refcount.hpp>
#include <vsip/core/expr/operations.hpp>
#include <vsip/core/expr/unary_block.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip/core/promote.hpp>

namespace vsip
{
namespace impl
{

/***********************************************************************
 Overloaded Operators
***********************************************************************/

/// Unary_operator_return_type encapsulates the inference
/// of a unary operator's return type from its arguments.
template <template <typename, typename> class View,
 	  typename Type,
	  typename Block,
          template <typename> class Operator,
	  typename Tag = typename Is_view_type<View<Type, Block> >::type>
struct Unary_operator_return_type
{
  typedef Type value_type;

  typedef expr::Unary<Operator, Block, true> const block_type;

  typedef typename ViewConversion<View, 
				  value_type, 
				  block_type>::const_view_type view_type;
};

template <template <typename, typename> class View,
 	  typename Type,
	  typename Block>
inline
typename Unary_operator_return_type<View, Type, Block, expr::op::Plus>::view_type
operator+(View<Type, Block> const& view)
{
  typedef Unary_operator_return_type<View, Type, Block, expr::op::Plus> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;
  return view_type(block_type(view.block()));
}

template <template <typename, typename> class View,
 	  typename Type,
	  typename Block>
inline
typename Unary_operator_return_type<View, Type, Block, expr::op::Minus>::view_type
operator-(View<Type, Block> const& view)
{
  typedef Unary_operator_return_type<View, Type, Block, expr::op::Minus> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;
  return view_type(block_type(view.block()));
}

} // namespace vsip::impl
} // namespace vsip

#endif

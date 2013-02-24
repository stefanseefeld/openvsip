/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/expr/evaluation.hpp
    @author  Stefan Seefeld
    @date    2009-06-08
    @brief   VSIPL++ Library: Expression evaluation
*/

#ifndef VSIP_CORE_EXPR_EVALUATION_HPP
#define VSIP_CORE_EXPR_EVALUATION_HPP

#include <vsip/core/expr/traversal.hpp>

namespace vsip_csl
{
namespace expr
{

template <typename BlockType> 
struct Evaluator 
{
  static void apply(BlockType const &) {}
};

template <typename BlockType> 
struct Evaluator<BlockType const> : Evaluator<BlockType> {};


template <template <typename> class Operation, typename Block>
struct Evaluator<Unary<Operation, Block, false> >
{
  typedef Unary<Operation, Block, false> block_type;

  static void apply(block_type const &b) { b.evaluate();}
};

template <template <typename, typename> class Operation,
	  typename Block0, typename Block1>
struct Evaluator<Binary<Operation, Block0, Block1, false> >
{
  typedef Binary<Operation, Block0, Block1, false> block_type;

  static void apply(block_type const &b) { b.evaluate();}
};

template <template <typename, typename, typename> class Operation,
	  typename Block0, typename Block1, typename Block2>
struct Evaluator<Ternary<Operation, Block0, Block1, Block2, false> >
{
  typedef Ternary<Operation, Block0, Block1, Block2, false> block_type;

  static void apply(block_type const &b) { b.evaluate();}
};

template <typename Block>
void
evaluate(Block const &block)
{
  Traversal<Evaluator, Block>::apply(block);
}

} // namespace vsip_csl::expr
} // namespace vsip_csl

#endif // VSIP_CORE_EXPR_EVALUATION_HPP

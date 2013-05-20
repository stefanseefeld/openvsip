//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_evaluate_hpp_
#define ovxx_expr_evaluate_hpp_

#include <ovxx/expr/traversal.hpp>

namespace ovxx
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

} // namespace ovxx::expr
} // namespace ovxx

#endif

//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_traversal_hpp_
#define ovxx_expr_traversal_hpp_

#include <ovxx/expr/generator.hpp>
#include <ovxx/expr/unary.hpp>
#include <ovxx/expr/binary.hpp>
#include <ovxx/expr/ternary.hpp>
// Forward-declare Vmmul rather than include its full definition
// to work around a circular dependency.
//#include <ovxx/expr/vmmul.hpp>
#include <ovxx/expr/transposed.hpp>
#include <ovxx/expr/subset.hpp>
#include <ovxx/expr/component.hpp>
#include <ovxx/expr/sliced.hpp>
#include <ovxx/expr/permuted.hpp>

namespace ovxx
{
namespace expr
{
template <dimension_type D, typename Block0, typename Block1>
class Vmmul;


/// Traversal traverses expression block trees,
/// applying the given functor to each node.
template <template <typename> class Functor,
          typename BlockType>
struct Traversal
{
  typedef BlockType block_type;
  static void apply(block_type const &b) { Functor<block_type>::apply(b);}
};

template <template <typename> class Functor,
          typename Block, template <typename> class Extractor>
struct Traversal<Functor, Component<Block, Extractor> >
{
  typedef Component<Block, Extractor> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.block());
    Functor<block_type>::apply(block);
  }
};
template <template <typename> class Functor,
          typename Block> 
struct Traversal<Functor, Subset<Block> >
{
  typedef Subset<Block> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.block());
    Functor<block_type>::apply(block);
  }
};
template <template <typename> class Functor,
          typename Block> 
struct Traversal<Functor, Transposed<Block> >
{
  typedef Transposed<Block> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.block());
    Functor<block_type>::apply(block);
  }
};
template <template <typename> class Functor,
          typename Block, typename Ordering> 
struct Traversal<Functor, Permuted<Block, Ordering> >
{
  typedef Permuted<Block, Ordering> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.block());
    Functor<block_type>::apply(block);
  }
};
template <template <typename> class Functor,
          typename Block, dimension_type D>
struct Traversal<Functor, Sliced<Block, D> >
{
  typedef Sliced<Block, D> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.block());
    Functor<block_type>::apply(block);
  }
};
template <template <typename> class Functor,
          typename Block, dimension_type D1, dimension_type D2>
struct Traversal<Functor, Sliced2<Block, D1, D2> >
{
  typedef Sliced2<Block, D1, D2> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.block());
    Functor<block_type>::apply(block);
  }
};

template <template <typename> class Functor,
          template <typename> class Operation,
          typename                  Block,
          bool                      Elementwise>
struct Traversal<Functor, Unary<Operation, Block, Elementwise> >
{
  typedef Unary<Operation, Block, Elementwise> block_type;

  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.arg());
    Functor<block_type>::apply(block);
  }
};

template <template <typename> class Functor,
          template <typename, typename> class Operation,
          typename LBlock,
          typename RBlock,
	  bool Elementwise>
struct Traversal<Functor,
                 Binary<Operation, LBlock, RBlock, Elementwise> >
{
  typedef Binary<Operation, LBlock, RBlock, Elementwise> block_type;

  static void apply(block_type const &block)
  {
    Traversal<Functor, LBlock>::apply(block.arg1());
    Traversal<Functor, RBlock>::apply(block.arg2());
    Functor<block_type>::apply(block);
  }
};

template <template <typename> class Functor,
          template <typename, typename, typename> class Operation,
          typename Block1,
          typename Block2,
          typename Block3,
	  bool Elementwise>
struct Traversal<Functor,
                 Ternary<Operation, Block1, Block2, Block3, Elementwise> >
{
  typedef Ternary<Operation, Block1, Block2, Block3, Elementwise> block_type;

  static void apply(block_type const &block)
  {
    Traversal<Functor, Block1>::apply(block.arg1());
    Traversal<Functor, Block2>::apply(block.arg2());
    Traversal<Functor, Block3>::apply(block.arg3());
    Functor<block_type>::apply(block);
  }
};

template <template <typename> class Functor,
          dimension_type D,
          typename       VectorBlock,
          typename       MatrixBlock>
struct Traversal<Functor,
                 expr::Vmmul<D, VectorBlock, MatrixBlock> >
{
  typedef expr::Vmmul<D, VectorBlock, MatrixBlock> block_type;

  static void apply(block_type const& block)
  {
    Traversal<Functor, VectorBlock>::apply(block.get_vblk());
    Traversal<Functor, MatrixBlock>::apply(block.get_mblk());
    Functor<block_type>::apply(block);
  }
};

} // namespace ovxx::expr
} // namespace ovxx

#endif

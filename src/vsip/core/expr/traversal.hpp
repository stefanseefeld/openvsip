//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_EXPR_TRAVERSAL_HPP
#define VSIP_CORE_EXPR_TRAVERSAL_HPP

#include <vsip/core/expr/unary_block.hpp>
#include <vsip/core/expr/binary_block.hpp>
#include <vsip/core/expr/ternary_block.hpp>
#include <vsip/core/expr/vmmul_block.hpp>

namespace vsip
{
namespace impl
{
template <typename B, template <typename> class E> class Component_block;
template <typename B> class Subset_block;
template <typename B> class Transposed_block;
template <typename B, typename O> class Permuted_block;
template <typename B, dimension_type D> class Sliced_block;
template <typename B, dimension_type D1, dimension_type D2> class Sliced2_block;
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace expr
{

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
struct Traversal<Functor, impl::Component_block<Block, Extractor> >
{
  typedef impl::Component_block<Block, Extractor> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.impl_block());
    Functor<block_type>::apply(block);
  }
};
template <template <typename> class Functor,
          typename Block> 
struct Traversal<Functor, impl::Subset_block<Block> >
{
  typedef impl::Subset_block<Block> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.impl_block());
    Functor<block_type>::apply(block);
  }
};
template <template <typename> class Functor,
          typename Block> 
struct Traversal<Functor, impl::Transposed_block<Block> >
{
  typedef impl::Transposed_block<Block> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.impl_block());
    Functor<block_type>::apply(block);
  }
};
template <template <typename> class Functor,
          typename Block, typename Ordering> 
struct Traversal<Functor, impl::Permuted_block<Block, Ordering> >
{
  typedef impl::Permuted_block<Block, Ordering> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.impl_block());
    Functor<block_type>::apply(block);
  }
};
template <template <typename> class Functor,
          typename Block, dimension_type D>
struct Traversal<Functor, impl::Sliced_block<Block, D> >
{
  typedef impl::Sliced_block<Block, D> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.impl_block());
    Functor<block_type>::apply(block);
  }
};
template <template <typename> class Functor,
          typename Block, dimension_type D1, dimension_type D2>
struct Traversal<Functor, impl::Sliced2_block<Block, D1, D2> >
{
  typedef impl::Sliced2_block<Block, D1, D2> block_type;
  static void apply(block_type const &block)
  {
    Traversal<Functor, Block>::apply(block.impl_block());
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

} // namespace vsip_csl::expr
} // namespace vsip_csl

#endif

/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   Example scale function using return block optimization.

#include <vsip/core/config.hpp>
#include <vsip/vector.hpp>
#include <vsip_csl/expr/functor.hpp>

namespace example
{
using namespace vsip;
using vsip_csl::expr::Unary;
using vsip_csl::expr::Unary_functor;

// Scale implements a call operator that scales its input
// argument, and returns it by reference.
template <typename ArgumentBlockType>
struct Scale : Unary_functor<ArgumentBlockType>
{
  Scale(ArgumentBlockType const &a, typename ArgumentBlockType::value_type s)
    : Unary_functor<ArgumentBlockType>(a), value(s) {}
  template <typename ResultBlockType>
  void apply(ResultBlockType &r) const
  {
    ArgumentBlockType const &a = this->arg();
    for (index_type i = 0; i != r.size(); ++i)
      r.put(i, a.get(i) * value);
  }

  typename ArgumentBlockType::value_type value;
};

// scale is a return-block optimised function returning an expression.
template <typename T, typename BlockType>
const_Vector<T, Unary<Scale, BlockType> const>
scale(const_Vector<T, BlockType> input, T value)
{
  typedef Unary<Scale, BlockType> block_type;
  Scale<BlockType> s(input.block(), value);
  block_type block(s);
  return const_Vector<T, Unary<Scale, BlockType> const>(block);
}
}

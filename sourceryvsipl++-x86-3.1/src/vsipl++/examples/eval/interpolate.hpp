/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   Example interpolate function using return block optimization.

#include <vsip/core/config.hpp>
#include <vsip/vector.hpp>
#include <vsip_csl/expr/functor.hpp>
#include <iostream>

namespace example
{
using namespace vsip;
using vsip_csl::View_block_storage;
using vsip_csl::expr::Unary;

// Interpolator models the (non-elementwise) UnaryFunctor concept.
// It generates a new block by interpolating an existing
// block. The size of the new block is specified.
template <typename ArgumentBlockType>
class Interpolator
{
public:
  typedef typename ArgumentBlockType::value_type value_type;
  typedef typename ArgumentBlockType::value_type result_type;
  typedef typename ArgumentBlockType::map_type map_type;
  static vsip::dimension_type const dim = ArgumentBlockType::dim;

  Interpolator(ArgumentBlockType const &a, Domain<ArgumentBlockType::dim> const &s)
    : argument_(a), size_(s) {}

  // Report the size of the new interpolated block
  length_type size() const { return size_.size();}
  length_type size(dimension_type b ATTRIBUTE_UNUSED, dimension_type d) const 
  {
    assert(b == ArgumentBlockType::dim);
    return size_[d].size();
  }
  map_type const &map() const { return argument_.map();}

  ArgumentBlockType const &arg() const { return argument_;}

  template <typename ResultBlockType>
  void apply(ResultBlockType &) const 
  {
    std::cout << "apply interpolation !" << std::endl;
    // interpolate 'argument' into 'result'
  }

private:
  typename View_block_storage<ArgumentBlockType>::expr_type argument_;
  Domain<ArgumentBlockType::dim> size_;
};

// interpolate is a return-block optimised function returning an expression.
template <typename T, typename BlockType>
const_Vector<T, Unary<Interpolator, BlockType> const>
interpolate(const_Vector<T, BlockType> arg, Domain<1> const &size) 
{
  typedef Unary<Interpolator, BlockType> block_type;
  Interpolator<BlockType> interpolator(arg.block(), size);
  block_type block(interpolator);
  return const_Vector<T, block_type const>(block);
}
}

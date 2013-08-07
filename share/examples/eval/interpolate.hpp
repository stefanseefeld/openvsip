/* Copyright (c) 2009, 2011 CodeSourcery, Inc.  All rights reserved. */

/* Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
         disclaimer in the documentation and/or other materials
         provided with the distribution.
       * Neither the name of CodeSourcery nor the names of its
         contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.

   THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */

/// Description
///   Example interpolate function using return block optimization.

#include <ovxx/config.hpp>
#include <vsip/vector.hpp>
#include <ovxx/expr/functor.hpp>
#include <iostream>

namespace example
{
using namespace vsip;
using ovxx::block_traits;
using ovxx::expr::Unary;

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
  length_type size(dimension_type b OVXX_UNUSED, dimension_type d) const 
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
  typename ovxx::block_traits<ArgumentBlockType>::expr_type argument_;
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

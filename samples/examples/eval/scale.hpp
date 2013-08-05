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

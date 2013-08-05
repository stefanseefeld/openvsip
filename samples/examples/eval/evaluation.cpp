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
///   Custom expression evaluator for a fused scale & interpolate operation.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip_csl/expr/eval.hpp>
#include "scale.hpp"
#include "interpolate.hpp"

using namespace example;


template <typename T, typename ResultBlockType, typename ArgumentBlockType>
void 
scaled_interpolate(Vector<T, ResultBlockType> result,
                   const_Vector<T, ArgumentBlockType> argument,
                   T scale, length_type new_size)
{
  length_type old_size = argument.size();
  float stretch = static_cast<float>(new_size)/old_size;
  for (index_type i = 0; i != new_size; ++i)
  {
    float pos = i / stretch;
    index_type j = static_cast<index_type>(pos);
    float alpha = pos - j;
    if (j + 1 == old_size)
      result.put(i, scale * (argument.get(j) * alpha));
    else
      result.put(i, scale * (argument.get(j) * alpha + argument.get(j + 1) * (1 - alpha)));
  }
}

namespace vsip_csl
{
namespace dispatcher
{

/// Custom evaluator for interpolate(scale(B, s), N)
template <typename LHS, typename ArgumentBlockType>
struct Evaluator<op::assign<1>, be::user,
		 void(LHS &,
                      expr::Unary<Interpolator, expr::Unary<Scale, ArgumentBlockType> const> const &)>
{
  typedef typename ArgumentBlockType::value_type value_type;
  typedef expr::Unary<Interpolator, expr::Unary<Scale, ArgumentBlockType> const> RHS;

  static char const *name() { return "be::user";}

  static bool const ct_valid = true;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    // rhs.arg() yields Unary<Scale, ArgumentBlockType>,
    // rhs.arg().arg() thus returns the terminal ArgumentBlockType block...
    ArgumentBlockType const &block = rhs.arg().arg();
    // ...and rhs.arg().operation() the Scale<ArgumentBlockType> functor.
    value_type scale = rhs.arg().operation().value;

    // rhs.operation() yields the Interpolator<Unary<Scale, ...> functor.
    length_type new_size(rhs.operation().size(1, 0));

    // wrap terminal blocks in views for convenience, and evaluate.
    Vector<value_type, LHS> result(lhs);
    const_Vector<value_type, ArgumentBlockType const> argument(block);
    scaled_interpolate(result, argument, scale, new_size);
  }
};

} // namespace vsip::impl
} // namespace vsip



int 
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  Vector<float> a(8, 2.);
  Vector<float> b = interpolate(scale(a, 2.f), 32);
}

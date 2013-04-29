/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

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

/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator expression support

#ifndef vsip_csl_pi_is_linear_expr_hpp_
#define vsip_csl_pi_is_linear_expr_hpp_

#include <vsip_csl/pi/call.hpp>
#include <vsip_csl/pi/expr.hpp>

namespace vsip_csl
{
namespace pi
{

template <typename T>
struct is_linear_expr { static bool const value = false;};

template <typename B, typename I, typename J, typename K>
struct is_linear_expr<Call<B, I, J, K> > { static bool const value = true;};

/// '+' operations are linear expressions if the operands are linear expressions.
template <typename A1, typename A2>
struct is_linear_expr<Binary<expr::op::Add, A1, A2> > 
{
  static bool const value =
    is_linear_expr<A1>::value &&
    is_linear_expr<A2>::value;
};

/// '-' operations are linear expressions if the operands are linear expressions.
template <typename A1, typename A2>
struct is_linear_expr<Binary<expr::op::Sub, A1, A2> > 
{
  static bool const value =
    is_linear_expr<A1>::value &&
    is_linear_expr<A2>::value;
};

/// For '*' and '/', only one of the operands may be an expression for
/// this to be linear.
template <template <typename, typename> class O, typename A1, typename A2>
struct is_linear_expr<Binary<O, A1, A2> > 
{
  static bool const value = !is_expr<A1>::value || !is_expr<A2>::value;
};


} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif

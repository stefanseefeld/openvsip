/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator expression support

#ifndef vsip_csl_pi_expr_hpp_
#define vsip_csl_pi_expr_hpp_

#include <vsip_csl/pi/iterator.hpp>

namespace vsip_csl
{
namespace pi
{

template <template <typename, typename> class O, typename A1, typename A2>
class Binary
{
public:
  Binary(A1 a1, A2 a2) : arg1_(a1), arg2_(a2) {}
  A1 const &arg1() const { return arg1_;}
  A2 const &arg2() const { return arg2_;}

private:
  A1 arg1_;
  A2 arg2_;
};

template <template <typename, typename> class O, typename A1, typename A2>
struct is_expr<Binary<O, A1, A2> > { static bool const value = true;};

template <typename A1, typename A2>
typename enable_if_c<is_expr<A1>::value || is_expr<A2>::value,
		     Binary<expr::op::Add, A1, A2> >::type
operator+(A1 a1, A2 a2)
{
  return Binary<expr::op::Add, A1, A2>(a1, a2);
}

template <typename A1, typename A2>
typename enable_if_c<is_expr<A1>::value || is_expr<A2>::value,
		     Binary<expr::op::Sub, A1, A2> >::type
operator-(A1 a1, A2 a2)
{
  return Binary<expr::op::Sub, A1, A2>(a1, a2);
}

template <typename A1, typename A2>
typename enable_if_c<is_expr<A1>::value || is_expr<A2>::value,
		     Binary<expr::op::Mult, A1, A2> >::type
operator*(A1 a1, A2 a2)
{
  return Binary<expr::op::Mult, A1, A2>(a1, a2);
}

template <typename A1, typename A2>
typename enable_if_c<is_expr<A1>::value || is_expr<A2>::value,
		     Binary<expr::op::Div, A1, A2> >::type
operator/(A1 a1, A2 a2)
{
  return Binary<expr::op::Div, A1, A2>(a1, a2);
}

} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif

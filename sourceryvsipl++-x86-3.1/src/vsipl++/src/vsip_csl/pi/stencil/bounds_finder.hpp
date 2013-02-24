/* Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Find kernel bounds.

#ifndef vsip_csl_pi_stencil_bounds_finder_hpp_
#define vsip_csl_pi_stencil_bounds_finder_hpp_

#include <vsip_csl/pi/expr.hpp>

namespace vsip_csl
{
namespace pi
{
namespace stencil
{

//
// Bound finder: Determine bounds for the stencil kernel.
//
struct Bounds
{
  Bounds() : x_prev(0), x_next(0), y_prev(0), y_next(0) {}
  vsip::length_type x_prev, x_next, y_prev, y_next;
};

template <typename E>
struct Bounds_finder
{
  // Traverse the expression E and extract the bounds of the kernel
  // to be constructed from it.
  static void apply(E e, Bounds& b);
};

template <typename S>
void find_bounds(S, Bounds &) {}

template <template <typename, typename> class O, typename L, typename R>
void find_bounds(Binary<O, L, R> e, Bounds &b)
{
  Bounds_finder<L>::apply(e.arg1(), b);
  Bounds_finder<R>::apply(e.arg2(), b);
}

template <typename V, typename I, typename J>
void find_bounds(Call<V, I, J> c, Bounds& b)
{
  find_y_bounds(c.i(), b);
  find_x_bounds(c.j(), b);
}

void find_y_bounds(Iterator<0>, Bounds &) {}
void find_x_bounds(Iterator<0>, Bounds &) {}

template <int I>
void find_y_bounds(Iterator<I>, Bounds &b)
{
  if (I > 0 && static_cast<length_type>(I) > b.y_next) b.y_next = I;
  else if (I < 0 && static_cast<length_type>(-I) > b.y_prev) b.y_prev = -I;
}

template <int I>
void find_x_bounds(Iterator<I>, Bounds &b)
{
  if (I > 0 && static_cast<length_type>(I) > b.x_next) b.x_next = I;
  else if (I < 0 && static_cast<length_type>(-I) > b.x_prev) b.x_prev = -I;
}

void find_y_bounds(Offset o, Bounds& b)
{
  if (o.i > 0 && static_cast<length_type>(o.i) > b.y_next) b.y_next = o.i;
  else if (o.i < 0 && static_cast<length_type>(-o.i) > b.y_prev) b.y_prev = -o.i;
}

void find_x_bounds(Offset o, Bounds& b)
{
  if (o.i > 0 && static_cast<length_type>(o.i) > b.x_next) b.x_next = o.i;
  else if (o.i < 0 && static_cast<length_type>(-o.i) > b.x_prev) b.x_prev = -o.i;
}

template <typename E>
void 
Bounds_finder<E>::apply(E e, Bounds &b)
{ find_bounds(e, b);}

} // namespace vsip_csl::pi::stencil
} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif

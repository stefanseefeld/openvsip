/* Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Stencil support classes.

#ifndef vsip_csl_pi_stencil_hpp_
#define vsip_csl_pi_stencil_hpp_

#include <vsip/matrix.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip_csl/pi/expr.hpp>
#include <vsip_csl/pi/stencil/boundary_factory.hpp>
#include <vsip_csl/pi/stencil/bounds_finder.hpp>
#include <vsip_csl/pi/stencil/kernel_builder.hpp>

namespace vsip_csl
{
namespace pi
{
namespace stencil
{

template <typename O> struct Inner_loop;

template <>
struct Inner_loop<vsip::row2_type>
{
  template <typename T1, typename Block1, typename T2, typename Block2,
            typename Op>
  static void apply(vsip::const_Matrix<T1, Block1> input,
                    vsip::Matrix<T2, Block2> output,
                    Op const &op,
                    vsip::length_type prev_y, vsip::length_type next_y,
                    vsip::length_type prev_x, vsip::length_type next_x)
  {
    for (vsip::length_type y = prev_y; y != input.size(0) - next_y; ++y)
      for (vsip::length_type x = prev_x; x != input.size(1) - next_x; ++x)
        output.put(y, x, op(input, y, x));
  }
};

template <>
struct Inner_loop<vsip::col2_type>
{
  template <typename T1, typename Block1, typename T2, typename Block2,
            typename Op>
  static void apply(vsip::const_Matrix<T1, Block1> input,
                    vsip::Matrix<T2, Block2> output,
                    Op const &op,
                    vsip::length_type prev_y, vsip::length_type next_y,
                    vsip::length_type prev_x, vsip::length_type next_x)
  {
    for (vsip::length_type x = prev_x; x != input.size(1) - next_x; ++x)
      for (vsip::length_type y = prev_y; y != input.size(0) - next_y; ++y)
        output.put(y, x, op(input, y, x));
  }
};

} // namespace vsip_csl::pi::stencil

template <typename T1, typename Block1, typename T2, typename Block2,
          typename Op>
void apply_stencil(vsip::const_Matrix<T1, Block1> input,
                   vsip::Matrix<T2, Block2> output,
                   Op const &op)
{
  using namespace stencil;
  using namespace vsip;

  index_type const prev_y = op.origin(0);
  index_type const prev_x = op.origin(1);
  index_type const next_y = op.size(0) - op.origin(0) - 1;
  index_type const next_x = op.size(1) - op.origin(1) - 1;

  // Compute the inner values.
  Inner_loop<typename vsip::impl::Block_layout<Block1>::order_type>::apply
    (input, output, op, prev_y, next_y, prev_x, next_x);

  if (prev_x)
  {
    // Compute the left boundary values.
    typedef Boundary_factory<Block1, left, Op> lb_factory;
    typedef typename lb_factory::view_type lb_view_type;
    lb_view_type lb = lb_factory::create(input.block(), op, zero);
    for (length_type y = prev_y; y != lb.size(0) - next_y; ++y)
      for (length_type x = 0; x != lb.size(1) - prev_x - next_x; ++x)
        output.put(y, x, op(lb, y, x + prev_x));
  }
  if (next_x)
  {
    // Compute the right boundary values.
    typedef Boundary_factory<Block1, right, Op> rb_factory;
    typedef typename rb_factory::view_type rb_view_type;
    rb_view_type rb = rb_factory::create(input.block(), op, zero);
    for (length_type y = prev_y; y != rb.size(0) - next_y; ++y)
      for (length_type x = 0; x != rb.size(1) - prev_x - next_x; ++x)
        output.put(y, output.size(1) - next_x + x,
                   op(rb, y, x + prev_x));
  }
  if (prev_y)
  {
    // Compute the top boundary values.
    typedef Boundary_factory<Block1, top, Op> tb_factory;
    typedef typename tb_factory::view_type tb_view_type;
    tb_view_type tb = tb_factory::create(input.block(), op, zero);
    for (length_type x = prev_x; x != tb.size(1) - next_x; ++x)
      for (length_type y = 0; y != tb.size(0) - prev_y - next_y; ++y)
        output.put(y, x, op(tb, y + prev_y, x));
  }
  if (next_y)
  {
    // Compute the bottom boundary values.
    typedef Boundary_factory<Block1, bottom, Op> bb_factory;
    typedef typename bb_factory::view_type bb_view_type;
    bb_view_type bb = bb_factory::create(input.block(), op, zero);
    for (length_type x = prev_x; x != bb.size(1) - next_x; ++x)
      for (length_type y = 0; y != bb.size(0) - prev_y - next_y; ++y)
        output.put(output.size(0) - next_y + y, x,
                   op(bb, y + prev_y, x));
  }
  if (prev_x && prev_y)
  {
    // Compute the top-left corner values.
    typedef Boundary_factory<Block1, top_left, Op> tlb_factory;
    typedef typename tlb_factory::view_type tlb_view_type;
    tlb_view_type tlb = tlb_factory::create(input.block(), op, zero);
    for (length_type y = 0; y != tlb.size(0) - prev_y - next_y; ++y)
      for (length_type x = 0; x != tlb.size(1) - prev_x - next_x; ++x)
        output.put(y, x, op(tlb, y + prev_y, x + prev_x));
  }
  if (next_y && prev_y)
  {
    // Compute the top-right corner values.
    typedef Boundary_factory<Block1, top_right, Op> trb_factory;
    typedef typename trb_factory::view_type trb_view_type;
    trb_view_type trb = trb_factory::create(input.block(), op, zero);
    for (length_type y = 0; y != trb.size(0) - prev_y - next_y; ++y)
      for (length_type x = 0; x != trb.size(1) - prev_x - next_x; ++x)
        output.put(y, output.size(1) - next_x + x,
                   op(trb, y + prev_y, x + prev_x));
  }
  if (prev_x && next_y)
  {
    // Compute the bottom-left corner values.
    typedef Boundary_factory<Block1, bottom_left, Op> blb_factory;
    typedef typename blb_factory::view_type blb_view_type;
    blb_view_type blb = blb_factory::create(input.block(), op, zero);
    for (length_type y = 0; y != blb.size(0) - prev_y - next_y; ++y)
      for (length_type x = 0; x != blb.size(1) - prev_x - next_x; ++x)
        output.put(output.size(0) - next_y + y, x,
                   op(blb, y + prev_y, x + prev_x));
  }
  if (next_y && next_x)
  {
    // Compute the bottom-right corner values.
    typedef Boundary_factory<Block1, bottom_right, Op> brb_factory;
    typedef typename brb_factory::view_type brb_view_type;
    brb_view_type brb = brb_factory::create(input.block(), op, zero);
    for (length_type y = 0; y != brb.size(0) - prev_y - next_y; ++y)
      for (length_type x = 0; x != brb.size(1) - prev_x - next_x; ++x)
        output.put(output.size(0) - next_y + y,
                   output.size(1) - next_x + x,
                   op(brb, y + prev_y, x + prev_x));
  }
}
  
namespace stencil
{

template <vsip::dimension_type D, typename T> struct Linear_expr_stencil;

template <typename T>
struct Linear_expr_stencil<2, T>
{
  Linear_expr_stencil(Kernel<T> &k) : kernel(k) {}

  vsip::length_type size(vsip::dimension_type d) const { return kernel.size(d);}
  vsip::index_type origin(vsip::dimension_type d) const { return kernel.origin(d);}

  template <typename V>
  typename V::value_type 
  operator()(V const &input, vsip::index_type y, vsip::index_type x) const
  {
    vsip::index_type const origin_y = kernel.origin(0);
    vsip::index_type const origin_x = kernel.origin(1);
    vsip::length_type const size_y = kernel.size(0);
    vsip::length_type const size_x = kernel.size(1);
    T result = T(0);
    for (vsip::index_type i = 0; i != size_y; ++i)
      for (vsip::index_type j = 0; j != size_x; ++j)
        result += kernel(i, j) * input.get(y + i - origin_y, x + j - origin_x);
    return result;
  }

  Kernel<T> &kernel;
};

template <typename V, typename I, typename J, typename RHS>
void assign(Call<V, I, J> lhs, RHS rhs)
{
  // First find the kernel bounds.
  Bounds bounds;
  Bounds_finder<RHS>::apply(rhs, bounds);

#if 0
  std::cout << "kernel bounds: "
            << bounds.y_prev << ' ' << bounds.y_next << ' '
            << bounds.x_prev << ' ' << bounds.x_next << std::endl;
#endif

  // Then construct the kernel and set its coefficients.
  Kernel<typename V::value_type> kernel(bounds.y_prev, bounds.x_prev,
                                        bounds.y_prev + bounds.y_next + 1,
                                        bounds.x_prev + bounds.x_next + 1);
  Kernel_builder<RHS, typename V::value_type>::apply(rhs, kernel);

#if 0
  std::cout << "kernel " << std::endl;
  for (int i = 0; i != kernel.size(0); ++i)
  {
    for (vsip::index_type j = 0; j != kernel.size(1); ++j)
      std::cout << kernel(i, j) << ' ';
    std::cout << '\n';
  }
  std::cout << std::endl;
#endif

  // Finally run it.
  Linear_expr_stencil<2, typename V::value_type> stencil(kernel);
  apply_stencil(rhs.view(), lhs.view(), stencil);
}

} // namespace vsip_csl::pi::stencil
} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif

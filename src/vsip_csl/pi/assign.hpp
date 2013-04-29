/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator expression assignment

#ifndef vsip_csl_pi_assign_hpp_
#define vsip_csl_pi_assign_hpp_

#include <vsip_csl/pi/call.hpp>
#include <vsip_csl/pi/expr.hpp>
#include <vsip_csl/pi/eval.hpp>
#include <vsip_csl/pi/is_linear_expr.hpp>
#include <vsip_csl/pi/is_stencil_expr.hpp>
#include <vsip_csl/pi/stencil.hpp>
#ifdef VSIP_IMPL_CBE_SDK
#  include <vsip/opt/cbe/ppu/pi_assign.hpp>
#endif
#include <stdexcept>

namespace vsip_csl
{
namespace dispatcher
{

/// Generic 1D expression
template <typename B, typename I, typename RHS>
struct Evaluator<op::pi_assign, be::generic,
		 void(pi::Call<B, I> &, RHS const &)>
{
  typedef pi::Call<B, I> LHS;

  static bool const ct_valid = pi::is_expr<RHS>::value;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs)
  {
    B &lhs_block = lhs.block();
    typedef pi::Evaluator<RHS> eval_type;

#pragma omp parallel for
    for (index_type i = 0; i < lhs_block.size(); ++i)
      lhs_block.put(i, eval_type::apply(rhs, i));
  }
};

/// Simple 1D elementwise assignment
template <typename B1, typename B2, typename I>
struct Evaluator<op::pi_assign, be::generic,
		 void(pi::Call<B1, I> &, pi::Call<B2, I> const &)>
{
  typedef pi::Call<B1, I> LHS;
  typedef pi::Call<B2, I> RHS;

  static bool const ct_valid = true;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs)
  {
    // Make sure both sides use the same iterator.
    if (&lhs.i() == &rhs.i())
      vsip::impl::assign<B1::dim>(lhs.block(), rhs.block());
    else
      VSIP_IMPL_THROW(std::invalid_argument("iterators don't match"));
  }
};

/// XXX Generic 2D expression (1 iterator, but 2D blocks)
template <typename B1, typename I, template <typename> class O, typename B2>
struct Evaluator<op::pi_assign, be::generic,
  void(pi::Call<B1, I, whole_domain_type> &,
       pi::Unary<O, pi::Call<B2, I, whole_domain_type> > const &)>
{
  typedef pi::Call<B1, I, whole_domain_type> LHS;
  typedef pi::Unary<O, pi::Call<B2, I, whole_domain_type> > RHS;

  static bool const ct_valid = true;//pi::is_expr<RHS>::value;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs)
  {
    for (index_type i = 0; i < lhs.block().size(2, 0); ++i)
    {
      typedef typename pi::Call<B1, I, whole_domain_type>::result_type result_type;
      typedef typename pi::Call<B2, I, whole_domain_type>::result_type argument_type;
      typedef O<argument_type> operation_type;
      operation_type const &op = rhs.operation();
      // lhs_block needs to be an lvalue, since it is passed down by non-const reference.
      result_type lhs_block = lhs.apply(i);
      op(lhs_block, rhs.arg().apply(i));
    }
  }
};

/// Simple 2D elementwise assignment
template <typename B1, typename B2, typename I, typename J>
struct Evaluator<op::pi_assign, be::generic,
		 void(pi::Call<B1, I, J> &, pi::Call<B2, I, J> const &),
		 typename enable_if_c<B1::dim == 2 && B2::dim == 2>::type>
{
  typedef pi::Call<B1, I, J> LHS;
  typedef pi::Call<B2, I, J> RHS;

  static bool const ct_valid = true;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs)
  {
    // Make sure both sides use the same iterator.
    if (&lhs.i() == &rhs.i() && &lhs.j() == &rhs.j())
      vsip::impl::assign<B1::dim>(lhs.block(), rhs.block());
    // Or it may be a transpose.
    else if (&lhs.i() == &rhs.j() && &lhs.j() == &rhs.i())
      vsip::impl::assign<B1::dim>(lhs.block(), rhs.trans().block());
    else
      VSIP_IMPL_THROW(std::invalid_argument("iterators don't match"));
  }
};

/// Linear 2D expressions are handled by stencils
template <typename B, typename I, typename J, typename RHS>
struct Evaluator<op::pi_assign, be::generic, 
		 void(pi::Call<B, I, J> &, RHS const &),
		 typename enable_if_c<pi::is_linear_expr<RHS>::value && B::dim == 2>::type>
{
  typedef pi::Call<B, I, J> LHS;

  static bool const ct_valid = true;
  static bool rt_valid(LHS &, RHS const &rhs)
  {
    typename pi::is_stencil_expr<RHS>::block_type const *dummy = 0;
    return pi::is_stencil_expr<RHS>::check(rhs, dummy);
  }
  static void exec(LHS &lhs, RHS const &rhs)
  {
    using namespace vsip_csl::pi;

    // First find the kernel bounds.
    stencil::Bounds bounds;
    stencil::Bounds_finder<RHS>::apply(rhs, bounds);
    // Then construct the kernel and set its coefficients.
    stencil::Kernel<typename B::value_type> 
      kernel(bounds.y_prev, bounds.x_prev,
	     bounds.y_prev + bounds.y_next + 1,
	     bounds.x_prev + bounds.x_next + 1);
    stencil::Kernel_builder<RHS, typename B::value_type>::apply(rhs, kernel);

    // Finally run it.
    stencil::Linear_expr_stencil<2, typename B::value_type> stencil(kernel);

    typedef typename pi::is_stencil_expr<RHS>::block_type rhs_block_type;
    typedef typename rhs_block_type::value_type rhs_value_type;
    typedef typename LHS::block_type lhs_block_type;
    typedef typename lhs_block_type::value_type lhs_value_type;

    rhs_block_type const *rhs_block = 0;
    pi::is_stencil_expr<RHS>::check(rhs, rhs_block);
    vsip::const_Matrix<rhs_value_type, rhs_block_type> 
      rhs_view(*const_cast<rhs_block_type*>(rhs_block));
    vsip::Matrix<lhs_value_type, lhs_block_type> lhs_view(lhs.block());
    vsip_csl::pi::apply_stencil(rhs_view, lhs_view, stencil);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

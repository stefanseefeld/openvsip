/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator signal functions support

#ifndef vsip_csl_pi_signal_hpp_
#define vsip_csl_pi_signal_hpp_

#include <vsip_csl/pi/expr.hpp>
#include <vsip_csl/pi/unary.hpp>
#include <vsip/signal.hpp>

namespace vsip_csl
{
namespace pi
{
namespace impl
{

/// Apply a Convolution object to a PI expression.
template <template <typename, typename> class V,
	  symmetry_type S, support_region_type R, typename T, unsigned N, alg_hint_type A>
struct Wrapper<Convolution<V, S, R, T, N, A> >
{
template <typename B>
class Operation
{
  typedef Convolution<V, S, R, T, N, A> operation_type;
public:
  typedef void result_type;

  Operation(operation_type &o) : operation_(o) {}
  operation_type &operation() const { return operation_;}

  template <typename LHS>
  void operator()(LHS &lhs, B const &arg_block) const
  {
    typedef typename vsip::impl::view_of<B>::type rhs_view_type;
    typedef typename vsip::impl::view_of<LHS>::type lhs_view_type;
    rhs_view_type rhs(const_cast<B&>(arg_block));
    // The Convolution call operator uses an (in, out) signature.
    operation_(rhs_view_type(rhs), lhs_view_type(lhs));
  }
private:
  // We need to store a non-const convolution reference here
  // since its call operator is non-const.
  operation_type &operation_;
};
};
} // namespace vsip_csl::pi::impl

} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif

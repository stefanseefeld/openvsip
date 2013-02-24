/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator unary expression support

#ifndef vsip_csl_pi_unary_hpp_
#define vsip_csl_pi_unary_hpp_

namespace vsip_csl
{
namespace pi
{
namespace impl
{

/// Apply a given function object to a PI expression.
/// Specialize for particular function object mappings.
template <typename F>
struct Wrapper
{
/// This wrapper also allows to wrap a stateful function object which may not
/// be copyable.
template <typename B>
class Operation
{
public:
  typedef typename B::value_type value_type;
  typedef typename vsip::impl::view_of<B>::type view_type;
  typedef typename F::result_type result_type;

  Operation() {}
  Operation(F const &f) : function_(f) {}

  result_type operator()(B const &block) const
  {
    view_type view(const_cast<B&>(block));
    return function_(view);
  }
  template <typename LHS>
  void operator()(LHS const &lhs, B const &block) const
  {
    view_type view(const_cast<B&>(block));
    function_(lhs, view);
  }
private:
  F function_;
};
};
} // namespace vsip_csl::pi::impl
} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif

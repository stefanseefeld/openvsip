/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator expression support

#ifndef vsip_csl_pi_foreach_hpp_
#define vsip_csl_pi_foreach_hpp_

#include <vsip_csl/pi/unary.hpp>
#include <vsip_csl/pi/signal.hpp>

namespace vsip_csl
{
namespace pi
{
/// foreach applies the given function object `f` to
/// each element in the domain that is being iterated over.
template <typename F, typename C>
typename enable_if<is_call<C>, Unary<impl::Wrapper<F>::template Operation, C> >::type
foreach(F &f, C const &call)
{ return Unary<impl::Wrapper<F>::template Operation, C>(f, call);}

/// foreach applies the given function type `F` to
/// each element in the domain that is being iterated over.
template <typename F, typename C>
typename enable_if<is_call<C>, Unary<impl::Wrapper<F>::template Operation, C> >::type
foreach(C const &call)
{ return Unary<impl::Wrapper<F>::template Operation, C>(call);}

} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif

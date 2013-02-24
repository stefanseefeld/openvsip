/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/dispatch.hpp
    @author  Don McCoy
    @date    2008-11-17
    @brief   VSIPL++ Library: Dispatcher harness basic definitions (see
               vsip/opt/dispatch.hpp for the actual dispatcher).
*/

#ifndef VSIP_CORE_DISPATCH_HPP
#define VSIP_CORE_DISPATCH_HPP

#include <vsip/core/dispatch_tags.hpp>
#include <vsip_csl/c++0x.hpp>

namespace vsip_csl
{
namespace dispatcher
{

/// Define the operation-specific Evaluator signature.
template <typename O, typename R = void> 
struct Signature
{
  // The default signature is useful for a compile-time check only,
  // as there are no arguments to inspect at runtime.
  typedef R(type)();
};


/// An Evaluator determines whether an Operation can be performed
/// with a particular backend.
///
/// Template parameters:
///   :O: Operation tag
///   :B: Backend tag
///   :S: Signature
///   :E: Enable (SFINAE) check.
template <typename O,
          typename B,
          typename S = typename Signature<O>::type,
	  typename E = void>
struct Evaluator
{
  static bool const ct_valid = false;
};


} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
namespace impl
{
namespace dispatcher = vsip_csl::dispatcher;
}
}

#endif // VSIP_CORE_DISPATCH_HPP

//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_EQUAL_HPP
#define VSIP_CORE_EQUAL_HPP

#include <vsip/core/fns_scalar.hpp>

namespace vsip
{
namespace impl
{

/// Compare two floating-point values for equality.
///
/// Algorithm from:
///    www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
template <typename T>
inline bool
almost_equal(T A, T B, T rel_epsilon = 1e-4, T abs_epsilon = 1e-6)
{
  if (vsip_csl::fn::mag(A - B) < abs_epsilon)
    return true;

  T relative_error;

  if (vsip_csl::fn::mag(B) > vsip_csl::fn::mag(A))
    relative_error = vsip_csl::fn::mag((A - B) / B);
  else
    relative_error = vsip_csl::fn::mag((B - A) / A);

  return (relative_error <= rel_epsilon);
}

} // namespace vsip::impl
} // namespace vsip

#endif

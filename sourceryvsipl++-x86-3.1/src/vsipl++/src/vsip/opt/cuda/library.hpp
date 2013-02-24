/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   CUDA library traits and initialization.

#ifndef vsip_opt_cuda_library_hpp_
#define vsip_opt_cuda_library_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>

namespace vsip
{
namespace impl
{
namespace cuda
{

void initialize(int& argc, char**&argv);

void finalize();

/// CUDA capabilities known at compile time are expressed as traits.
template <typename T>
struct Traits
{
  static bool const valid = false;
};

template <>
struct Traits<float>
{
  static bool const valid = true;
  static char const trans = 't';
};

template <>
struct Traits<double>
{
  static bool const valid = false;
  static char const trans = 't';
};

template <>
struct Traits<std::complex<float> >
{
  static bool const valid = true;
  static char const trans = 'c';
};

template <>
struct Traits<std::complex<double> >
{
  static bool const valid = false;
  static char const trans = 'c';
};

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif

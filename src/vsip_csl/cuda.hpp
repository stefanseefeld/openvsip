/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   CUDA API

#ifndef vsip_csl_cuda_hpp_
#define vsip_csl_cuda_hpp_

#include <vsip/opt/cuda/dda.hpp>
#include <vsip/opt/cuda/device.hpp>
#include <vsip/opt/cuda/module.hpp>

namespace vsip_csl
{
namespace cuda
{
using vsip::impl::cuda::Device;
using vsip::impl::cuda::Function;
using vsip::impl::cuda::Module;
using vsip::impl::cuda::num_devices;
using vsip::impl::cuda::get_device;
using vsip::impl::cuda::set_device;

namespace dda
{

using vsip::dda::sync_policy;
using vsip::dda::in;
using vsip::dda::out;
using vsip::dda::inout;
using vsip::impl::cuda::dda::Data;

} // namespace vsip_csl::cuda::dda
} // namespace vsip_csl::cuda
} // namespace vsip_csl

#endif

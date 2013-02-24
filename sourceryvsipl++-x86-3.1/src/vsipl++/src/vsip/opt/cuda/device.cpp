/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#include <vsip/opt/cuda/device.hpp>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>

namespace vsip
{
namespace impl
{
namespace cuda
{

void set_device(int device)
{
  cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess)
    VSIP_IMPL_THROW(std::runtime_error("set_device"));
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

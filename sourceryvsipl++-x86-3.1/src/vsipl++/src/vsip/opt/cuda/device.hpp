/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   CUDA device access API

#ifndef vsip_opt_cuda_device_hpp_
#define vsip_opt_cuda_device_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/opt/cuda/exception.hpp>
#include <cuda.h>

namespace vsip
{
namespace impl
{
namespace cuda
{

class Device
{
public:
  Device(CUdevice dev) : device_(dev) {}

  std::string name()
  {
    char buffer[1024];
    VSIP_IMPL_CUDA_CHECK_RESULT(cuDeviceGetName, (buffer, sizeof(buffer), device_));
    return buffer;
  }

  void compute_capability(int &major, int &minor) const
  {
    VSIP_IMPL_CUDA_CHECK_RESULT(cuDeviceComputeCapability, (&major, &minor, device_));
  }

  unsigned int total_memory()
  {
    unsigned int bytes;
    VSIP_IMPL_CUDA_CHECK_RESULT(cuDeviceTotalMem, (&bytes, device_));
    return bytes;
  }

  int get_attribute(CUdevice_attribute attr) const
  {
    int result;
    VSIP_IMPL_CUDA_CHECK_RESULT(cuDeviceGetAttribute, (&result, attr, device_));
    return result;
  }

  bool operator==(Device const &other) const { return device_ == other.device_;}

  operator CUdevice() const { return device_;}

private:
  CUdevice device_;
};

inline int num_devices()
{
  int result;
  VSIP_IMPL_CUDA_CHECK_RESULT(cuDeviceGetCount, (&result));
  return result;
}

inline Device get_device(int ordinal)
{
  CUdevice result;
  VSIP_IMPL_CUDA_CHECK_RESULT(cuDeviceGet, (&result, ordinal));
  return Device(result);
}

inline Device get_device()
{
  CUdevice result;
  VSIP_IMPL_CUDA_CHECK_RESULT(cuCtxGetDevice, (&result));
  return Device(result);
}

void set_device(int);

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif

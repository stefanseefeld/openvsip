//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/opencl/device.hpp>
#include <ovxx/opencl/platform.hpp>

namespace ovxx
{
namespace opencl
{
device default_device()
{
  static device dev;
  if (!dev.id())
  {
    platform pl = default_platform();
    std::vector<device> devices = pl.devices(device::accelerator);
    if (devices.size()) dev = devices[0];
    else // no accelerators found, try all
    {
      std::vector<device> devices = pl.devices(device::all);
      if (devices.size()) dev = devices[0];
      else OVXX_DO_THROW(std::runtime_error("No OpenCL devices found."));
    }
  }
  return dev;
}

} // namespace ovxx
} // namespace opencl

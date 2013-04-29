/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <boost/python.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip_csl/cuda.hpp>

namespace bpl = boost::python;
using namespace vsip_csl::cuda;

bpl::tuple compute_capability(Device const &d)
{
  int major, minor;
  d.compute_capability(major, minor);
  return bpl::make_tuple(major, minor);
}

BOOST_PYTHON_MODULE(device)
{
  bpl::class_<Device> device("Device", bpl::no_init);
  device.def("name", &Device::name);
  device.def("compute_capability", compute_capability);
  device.def("total_memory", &Device::total_memory);

  bpl::def("num_devices", num_devices);
  bpl::def("get_device", (Device(*)(int))get_device);
  bpl::def("get_device", (Device(*)())get_device);
  bpl::def("set_device", set_device);
}

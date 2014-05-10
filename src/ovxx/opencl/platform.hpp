//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_platform_hpp_
#define ovxx_opencl_platform_hpp_

#include <ovxx/opencl/exception.hpp>
#include <ovxx/opencl/device.hpp>
#include <vector>
#include <sstream>
#include <string>
#include <iterator>

namespace ovxx
{
namespace opencl
{
class platform
{
public:
  static std::vector<platform> platforms()
  {
    cl_uint nplatforms;
    OVXX_OPENCL_CHECK_RESULT(clGetPlatformIDs, (0, 0, &nplatforms));
    std::vector<platform> platforms;
  
    if (nplatforms)
    {
      cl_platform_id *ids = new cl_platform_id[nplatforms];
      OVXX_OPENCL_CHECK_RESULT(clGetPlatformIDs, (nplatforms, ids, 0));
      for (unsigned i = 0; i < nplatforms; ++i)
	platforms.push_back(platform(ids[i]));
      delete[] ids;
    }
    return platforms;
  }

  platform(cl_platform_id id) : id_(id) {}
  std::string profile() const { return info(CL_PLATFORM_PROFILE);}
  std::string version() const { return info(CL_PLATFORM_VERSION);}
  std::string vendor() const { return info(CL_PLATFORM_VENDOR);}
  std::string name() const { return info(CL_PLATFORM_NAME);}
  std::vector<std::string> extensions() const
  {
    std::string ext = info(CL_PLATFORM_EXTENSIONS);
    std::istringstream buf(ext);
    std::istream_iterator<std::string> beg(buf), end;
    return std::vector<std::string>(beg, end);
  }
  std::vector<device> devices(device::type type = device::all)
  {
    cl_uint ndevices = 0;
    int status = clGetDeviceIDs(id_, type, 0, 0, &ndevices);
    // a status of -1 is not an error.
    if (status < -1) OVXX_DO_THROW(exception("clGetDeviceIDs", status));
    std::vector<device> devices;
    if (ndevices)
    {
      cl_device_id *ids = new cl_device_id[ndevices];
      OVXX_OPENCL_CHECK_RESULT(clGetDeviceIDs, (id_, type, ndevices, ids, 0));
      for (unsigned i = 0; i < ndevices; ++i)
	devices.push_back(device(ids[i]));
      delete[] ids;
    }
    return devices;
  }

  operator cl_platform_id() { return id_;}
private:
  std::string info(cl_platform_info i) const
  {
    char name[1024];
    OVXX_OPENCL_CHECK_RESULT(clGetPlatformInfo, (id_, i, sizeof(name), name, 0));
    return std::string(name);
  }
  cl_platform_id id_;
};

} // namespace ovxx::opencl
} // namespace ovxx

#endif

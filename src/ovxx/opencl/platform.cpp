//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/opencl/platform.hpp>
#include <cstdlib>

namespace ovxx
{
namespace opencl
{

// Create a default OpenCL platform. The first one reported is
// typically "Default", which ought to work everywhere. However,
// a case has been detected where it doesn't, so we allow an 
// environment variable to override the chosen index.
platform default_platform()
{
  size_t idx = 0;
  char *ovxx_opencl_platform = std::getenv("OVXX_OPENCL_PLATFORM");
  if (ovxx_opencl_platform)
  {
    std::istringstream iss(ovxx_opencl_platform);
    iss >> idx;
    if (!iss)
      OVXX_DO_THROW(std::runtime_error("Unable to extract index value from OVXX_OPENCL_PLATFORM envvar."));
  }

  cl_uint nplatforms;
  OVXX_OPENCL_CHECK_RESULT(clGetPlatformIDs, (0, 0, &nplatforms));
  if (!nplatforms)
    OVXX_DO_THROW(std::runtime_error("No OpenCL platform detected."));
  else if (idx >= nplatforms)
    OVXX_DO_THROW(std::invalid_argument("Invalid OpenCL platform index."));
  cl_platform_id *ids = new cl_platform_id[nplatforms];
  OVXX_OPENCL_CHECK_RESULT(clGetPlatformIDs, (nplatforms, ids, 0));
  platform pl(ids[idx]);
  delete[] ids;
  return pl;
}

} // namespace ovxx::opencl
} // namespace ovxx

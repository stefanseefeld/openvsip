//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_program_hpp_
#define ovxx_opencl_program_hpp_

#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/opencl/exception.hpp>
#include <ovxx/opencl/device.hpp>
#include <ovxx/opencl/kernel.hpp>
#include <vector>

namespace ovxx
{
namespace opencl
{
class context;

class program : ovxx::detail::noncopyable
{
  friend class context;
public:
  ~program() { OVXX_OPENCL_CHECK_RESULT(clReleaseProgram, (impl_));}

  void build(std::vector<device> const &d)
  {
    cl_device_id ids_[8];
    cl_device_id *ids = 0;
    if (d.size() > 8)
      ids = new cl_device_id[d.size()];
    else
      ids = ids_;
    std::copy(d.begin(), d.end(), ids);
    OVXX_OPENCL_CHECK_RESULT(clBuildProgram,
      (impl_, d.size(), ids, 0, 0, 0));
    if (d.size() > 8)
      delete [] ids;
  }
  kernel *create_kernel(std::string const &name)
  {
    return new kernel(impl_, name);
  }
private:
  program(cl_context c, std::string const &source)
  {
    cl_int status;
    char const *sources[] = {source.data()};
    size_t size[] = { source.size()};
    impl_ = clCreateProgramWithSource(c, 1, sources, size, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateProgramWithSource", status));
  }
  cl_program impl_;
};

} // namespace ovxx::opencl
} // namespace ovxx

#endif

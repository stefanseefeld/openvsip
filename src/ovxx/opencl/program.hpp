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
  program() : impl_(0) {}
  explicit program(cl_program p, bool retain = true)
    : impl_(p)
  {
    if (impl_ && retain) OVXX_OPENCL_CHECK_RESULT(clRetainProgram, (impl_));
  }
  program(program const &other)
    : impl_(other.impl_)
  {
    if (impl_) OVXX_OPENCL_CHECK_RESULT(clRetainProgram, (impl_));
  }
  program &operator=(program const &other)
  {
    if (this != &other)
    {
      if (impl_) OVXX_OPENCL_CHECK_RESULT(clReleaseProgram, (impl_));
      impl_ = other.impl_;
      if (impl_) OVXX_OPENCL_CHECK_RESULT(clRetainProgram, (impl_));
    }
    return *this;
  }
  ~program() { OVXX_OPENCL_CHECK_RESULT(clReleaseProgram, (impl_));}

  static program create_with_source(context const &c, std::string const &source)
  {
    cl_int status;
    char const *sources[] = {source.data()};
    size_t size[] = { source.size()};
    cl_program p = clCreateProgramWithSource(c.get(), 1, sources, size, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateProgramWithSource", status));
    return program(p, false);
  }

  void build(std::vector<device> const &d)
  {
    std::vector<cl_device_id> ids(d.size());
    std::transform(d.begin(), d.end(), ids.begin(), std::mem_fun_ref(&device::id));
    OVXX_OPENCL_CHECK_RESULT(clBuildProgram, 
      (impl_, d.size(), &*ids.begin(), 0, 0, 0));
  }
  kernel create_kernel(std::string const &name)
  {
    return kernel(impl_, name);
  }
private:
  cl_program impl_;
};

} // namespace ovxx::opencl
} // namespace ovxx

#endif

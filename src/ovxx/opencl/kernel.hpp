//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_kernel_hpp_
#define ovxx_opencl_kernel_hpp_

#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/opencl/exception.hpp>
#include <ovxx/opencl/buffer.hpp>
#include <vector>

namespace ovxx
{
namespace opencl
{
class program;

class kernel : ovxx::detail::noncopyable
{
  friend class program;
public:
  ~kernel() { OVXX_OPENCL_CHECK_RESULT(clReleaseKernel, (impl_));}

  operator cl_kernel() { return impl_;}

  void set_arg(int i, buffer &b)
  {
    OVXX_OPENCL_CHECK_RESULT(clSetKernelArg, 
      (impl_, i, sizeof(cl_mem), &static_cast<cl_mem &>(b)));
  }

private:
  kernel(cl_program p, std::string const &source)
  {
    cl_int status;
    impl_ = clCreateKernel(p, source.c_str(), &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateKernel", status));
  }
  cl_kernel impl_;
};

} // namespace ovxx::opencl
} // namespace ovxx

#endif

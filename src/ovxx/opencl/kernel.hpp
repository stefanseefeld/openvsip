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
#include <ovxx/opencl/command_queue.hpp>

namespace ovxx
{
namespace opencl
{
class program;

class kernel : ovxx::detail::noncopyable
{
  friend class program;
public:
  kernel() : impl_(0) {}
  explicit kernel(cl_kernel k, bool retain = true)
    : impl_(k)
  {
    if (k && retain) OVXX_OPENCL_CHECK_RESULT(clRetainKernel, (impl_));
  }
  kernel(kernel const &other)
    : impl_(other.impl_)
  {
    if (impl_) OVXX_OPENCL_CHECK_RESULT(clRetainKernel, (impl_));
  }
  ~kernel() { OVXX_OPENCL_CHECK_RESULT(clReleaseKernel, (impl_));}
  kernel &operator=(kernel const &other)
  {
    if (this != &other)
    {
      if (impl_) OVXX_OPENCL_CHECK_RESULT(clReleaseKernel, (impl_));
      impl_ = other.impl_;
      if (impl_) OVXX_OPENCL_CHECK_RESULT(clRetainKernel, (impl_));
    }
    return *this;
  }
  cl_kernel get() const { return impl_;}

  // Doing the following with varargs doesn't work,
  // because we need the compiler to force type conversion (buffer->cl_mem, notably).
  void exec(command_queue &q, size_t s, buffer a)
  {
    set_arg(0, a);
    exec(q, s);
  }
  void exec(command_queue &q, size_t s, buffer a0, buffer a1)
  {
    set_arg(0, a0);
    set_arg(1, a1);
    exec(q, s);
  }
  void exec(command_queue &q, size_t s, buffer a0, buffer a1, buffer a2)
  {
    set_arg(0, a0);
    set_arg(1, a1);
    set_arg(2, a2);
    exec(q, s);
  }
  void exec(command_queue &q, size_t s, buffer a0, buffer a1, buffer a2, buffer a3)
  {
    set_arg(0, a0);
    set_arg(1, a1);
    set_arg(2, a2);
    set_arg(3, a3);
    exec(q, s);
  }

  void exec(command_queue &queue, size_t size)
  {
    size_t global_work_size[1] = {size};
    OVXX_OPENCL_CHECK_RESULT(clEnqueueNDRangeKernel,
       (queue.get(), get(), 1, 0, global_work_size, 0, 0, 0, 0));
  }

  void set_arg(int i, buffer m)
  {
    OVXX_OPENCL_CHECK_RESULT(clSetKernelArg, (impl_, i, sizeof(cl_mem), &m.get()));
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

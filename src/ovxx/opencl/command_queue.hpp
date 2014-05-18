//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_command_queue_hpp_
#define ovxx_opencl_command_queue_hpp_

#include <ovxx/opencl/exception.hpp>
#include <ovxx/opencl/context.hpp>
#include <ovxx/opencl/device.hpp>
#include <ovxx/opencl/buffer.hpp>

namespace ovxx
{
namespace opencl
{
class command_queue
{
public:
  command_queue() : impl_(0) {}
  explicit command_queue(cl_command_queue q, bool retain = true)
    : impl_(q)
  {
    if (impl_ && retain) OVXX_OPENCL_CHECK_RESULT(clRetainCommandQueue, (impl_));
  }
  command_queue(opencl::context const &c, opencl::device const &d,
		cl_command_queue_properties props = 0)
  {
    cl_int status;
    impl_ = clCreateCommandQueue(c.get(), d.id(), props, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateCommandQueue", status));
  }
  command_queue(command_queue const &other)
    : impl_(other.impl_)
  {
    if (impl_) OVXX_OPENCL_CHECK_RESULT(clRetainCommandQueue, (impl_));
  }
  ~command_queue() { if (impl_) OVXX_OPENCL_CHECK_RESULT(clReleaseCommandQueue, (impl_));}
  command_queue &operator=(command_queue const &other)
  {
    if (this != &other)
    {
      if (impl_) OVXX_OPENCL_CHECK_RESULT(clReleaseCommandQueue, (impl_));
      impl_ = other.impl_;
      if (impl_) OVXX_OPENCL_CHECK_RESULT(clRetainCommandQueue, (impl_));
    }
    return *this;
  }
  cl_command_queue &get() { return impl_;}

  opencl::device device() const
  { return opencl::device(info<cl_device_id>(CL_QUEUE_DEVICE));}
  opencl::context context() const
  { return opencl::context(info<cl_context>(CL_QUEUE_CONTEXT));}

  template <typename T>
  void write(T const *data, buffer &b, size_t s)
  {
    OVXX_TRACE("OpenCL copy %d bytes to device", s*sizeof(T));
    OVXX_OPENCL_CHECK_RESULT(clEnqueueWriteBuffer,
      (get(), b.get(), CL_TRUE, 0, s*sizeof(T), data, 0, 0, 0));
  }
  template <typename T>
  void read(buffer const &b, T *data, size_t s)
  {
    OVXX_TRACE("OpenCL copy %d bytes from device", s*sizeof(T));
    OVXX_OPENCL_CHECK_RESULT(clEnqueueReadBuffer,
      (get(), b.get(), CL_TRUE, 0, s*sizeof(T), data, 0, 0, 0));
  }

private:

  template <typename T>
  T info(cl_command_queue_info i) const
  {
    T param;
    OVXX_OPENCL_CHECK_RESULT(clGetCommandQueueInfo,
      (impl_, i, sizeof(param), &param, 0));
    return param;
  }

  cl_command_queue impl_;
};

command_queue default_queue();

} // namespace ovxx::opencl
} // namespace ovxx

#endif

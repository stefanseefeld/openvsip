//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_context_hpp_
#define ovxx_opencl_context_hpp_

#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/opencl/exception.hpp>
#include <ovxx/opencl/device.hpp>
#include <ovxx/opencl/command_queue.hpp>
#include <ovxx/opencl/buffer.hpp>
#include <ovxx/opencl/program.hpp>
#include <vector>
#include <iostream>

namespace ovxx
{
namespace opencl
{
class context : ovxx::detail::noncopyable
{
public:
  context(intptr_t *props, device::type t)
  {
    cl_int status;
    impl_ = clCreateContextFromType(props, t, context::callback, this, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateContextFromType", status));
  }
  context(std::vector<device> const &d)
  {
    cl_device_id ids_[8];
    cl_device_id *ids = 0;
    if (d.size() > 8)
      ids = new cl_device_id[d.size()];
    else
      ids = ids_;
    std::copy(d.begin(), d.end(), ids);
    cl_int status;
    impl_ = clCreateContext(0, d.size(), ids,
			    context::callback, this, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateContext", status));
    if (d.size() > 8)
      delete [] ids;
  }
  ~context() { OVXX_OPENCL_CHECK_RESULT(clReleaseContext, (impl_));}
  std::vector<device> devices() const
  {
    size_t bytes;
    OVXX_OPENCL_CHECK_RESULT(clGetContextInfo, (impl_, CL_CONTEXT_DEVICES, 0, 0, &bytes));
    size_t ndevices = bytes / sizeof(cl_device_id);
    std::vector<device> devices;
    cl_device_id *ids = new cl_device_id[ndevices];
    OVXX_OPENCL_CHECK_RESULT(clGetContextInfo,
      (impl_, CL_CONTEXT_DEVICES, ndevices*sizeof(cl_device_id), ids, 0));
    for (unsigned i = 0; i < ndevices; ++i)
      devices.push_back(device(ids[i]));
    delete[] ids;
    return devices;
  }
  command_queue *create_queue(device &d)
  {
    return new command_queue(impl_, d);
  }
  program *create_program(std::string const &src)
  {
    return new program(impl_, src);
  }
  template <typename T>
  buffer *create_buffer(int flags, T *data, size_t len)
  {
    return new buffer(impl_, flags, data, len);
  }
private:
  void error(char const *msg)
  {
    std::cerr << "Error : " << msg << std::endl;
  }

  static void callback(char const *msg, void const *, size_t, void *closure)
  {
    static_cast<context*>(closure)->error(msg);
  }
  cl_context impl_;
};

} // namespace ovxx::opencl
} // namespace ovxx

#endif

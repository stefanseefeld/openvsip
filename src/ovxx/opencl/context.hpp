//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_context_hpp_
#define ovxx_opencl_context_hpp_

#include <ovxx/opencl/exception.hpp>
#include <ovxx/opencl/device.hpp>
#include <vector>
#include <algorithm>

namespace ovxx
{
namespace opencl
{
class context
{
public:
  context() : impl_(0) {}
  explicit context(cl_context c, bool retain = true)
    : impl_(c)
  {
    if (c && retain) OVXX_OPENCL_CHECK_RESULT(clRetainContext, (impl_));
  }
  context(device const &d, cl_context_properties *props = 0)
  {
    OVXX_PRECONDITION(d.id() != 0);
    cl_device_id id = d.id();
    cl_int status;
    impl_ = clCreateContext(props, 1, &id, context::callback, this, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateContext", status));
  }
  context(cl_device_type t, cl_context_properties *props = 0)
  {
    cl_int status;
    impl_ = clCreateContextFromType(props, t, context::callback, this, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateContextFromType", status));
  }
  context(std::vector<device> const &d, cl_context_properties *props = 0)
  {
    std::vector<cl_device_id> ids(d.size());
    std::transform(d.begin(), d.end(), ids.begin(), std::mem_fun_ref(&device::id));
    cl_int status;
    impl_ = clCreateContext(props, d.size(), &*ids.begin(),
			    context::callback, this, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateContext", status));
  }
  context(context const &c) : impl_(c.impl_)
  {
    if (impl_) OVXX_OPENCL_CHECK_RESULT(clRetainContext, (impl_));
  }
  ~context() { if (impl_) OVXX_OPENCL_CHECK_RESULT(clReleaseContext, (impl_));}
  context &operator=(context const &other)
  {
    if (this != &other)
    {
      if (impl_) OVXX_OPENCL_CHECK_RESULT(clReleaseContext, (impl_));
      impl_ = other.impl_;
      if (impl_) OVXX_OPENCL_CHECK_RESULT(clRetainContext, (impl_));
    }
    return *this;
  }
  bool operator==(context const &other) const { return impl_ == other.impl_;}
  cl_context get() const { return impl_;}
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
private:
  void error(char const *msg)
  {
    OVXX_DO_THROW(std::runtime_error(msg));
  }

  static void callback(char const *msg, void const *, size_t, void *closure)
  {
    static_cast<context*>(closure)->error(msg);
  }
  cl_context impl_;
};

context default_context();

} // namespace ovxx::opencl
} // namespace ovxx

#endif

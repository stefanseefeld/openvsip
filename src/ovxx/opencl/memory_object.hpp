//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_memory_object_hpp_
#define ovxx_opencl_memory_object_hpp_

#include <ovxx/opencl/exception.hpp>

namespace ovxx
{
namespace opencl
{

class memory_object
{
public:
  enum mem_flags 
  {
    read_write = CL_MEM_READ_WRITE,
    read = CL_MEM_READ_ONLY,
    write = CL_MEM_WRITE_ONLY,
  };

  cl_mem &get() const { return const_cast<cl_mem &>(impl_);}
  length_type size() const { return info<size_t>(CL_MEM_SIZE);}
protected:
  memory_object() : impl_(0) {}
  explicit memory_object(cl_mem m, bool retain = true)
    : impl_(m)
  {
    if (impl_ && retain) OVXX_OPENCL_CHECK_RESULT(clRetainMemObject, (impl_));
  }
  memory_object(memory_object const &other)
    : impl_(other.impl_)
  {
    if (impl_) OVXX_OPENCL_CHECK_RESULT(clRetainMemObject, (impl_));
  }
  ~memory_object()
  {
    if (impl_) OVXX_OPENCL_CHECK_RESULT(clReleaseMemObject, (impl_));
  }
  memory_object &operator=(memory_object const &other)
  {
    if (this != &other)
    {
    if (impl_) OVXX_OPENCL_CHECK_RESULT(clReleaseMemObject, (impl_));
    impl_ = other.impl_;
    if (impl_) OVXX_OPENCL_CHECK_RESULT(clRetainMemObject, (impl_));
    }
    return *this;
  }
  bool operator==(memory_object const &other) const
  {
    return impl_ == other.impl_;
  }

  cl_mem impl_;
private:
  template <typename T>
  T info(cl_mem_info i) const
  {
    T param;
    OVXX_OPENCL_CHECK_RESULT(clGetMemObjectInfo,
      (impl_, i, sizeof(param), &param, 0));
    return param;
  }
};

} // namespace ovxx::opencl
} // namespace ovxx

#endif

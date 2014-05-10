//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_buffer_hpp_
#define ovxx_opencl_buffer_hpp_

#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/opencl/exception.hpp>
#include <ovxx/opencl/device.hpp>
#include <vector>

namespace ovxx
{
namespace opencl
{
class context;

class buffer : ovxx::detail::noncopyable
{
  friend class context;
public:
  ~buffer() { OVXX_OPENCL_CHECK_RESULT(clReleaseMemObject, (impl_));}

  operator cl_mem &() { return impl_;}
private:
  template <typename T>
  buffer(cl_context c, int flags, T *data, size_t len)
  {
    cl_int status;
    impl_ = clCreateBuffer(c, flags, len*sizeof(T), data, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateBuffer", status));
  }
  cl_mem impl_;
};

} // namespace ovxx::opencl
} // namespace ovxx

#endif

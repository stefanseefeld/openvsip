//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_command_queue_hpp_
#define ovxx_opencl_command_queue_hpp_

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

class command_queue : ovxx::detail::noncopyable
{
  friend class context;
public:
  ~command_queue() { OVXX_OPENCL_CHECK_RESULT(clReleaseCommandQueue, (impl_));}

  void push_back(kernel &k, size_t s)
  {
    size_t global_work_size[1] = {s};
    OVXX_OPENCL_CHECK_RESULT(clEnqueueNDRangeKernel,
      (impl_, k, 1, 0, global_work_size, 0, 0, 0, 0));
  }
  template <typename T>
  void push_back(buffer &b, T *data, size_t s)
  {
    OVXX_OPENCL_CHECK_RESULT(clEnqueueReadBuffer,
      (impl_, b, CL_TRUE, 0, s*sizeof(T), data, 0, 0, 0));
  }
private:
  command_queue(cl_context c, device const &d)
  {
    cl_int status;
    impl_ = clCreateCommandQueue(c, d, 0, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateCommandQueue", status));
  }
    cl_command_queue impl_;
};
  
} // namespace ovxx::opencl
} // namespace ovxx

#endif

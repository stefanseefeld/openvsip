//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_buffer_hpp_
#define ovxx_opencl_buffer_hpp_

#include <ovxx/opencl/memory_object.hpp>
#include <ovxx/opencl/context.hpp>

namespace ovxx
{
namespace opencl
{
class command_queue;

class buffer : public memory_object
{
public:
  buffer() : memory_object() {}
  explicit buffer(cl_mem m, bool retain = true) : memory_object(m, retain) {}
  buffer(context const &c, size_t size, cl_mem_flags flags = read_write)
  {
    cl_int status;
    impl_ = clCreateBuffer(c.get(), flags, size, 0, &status);
    if (status < 0)
      OVXX_DO_THROW(exception("clCreateBuffer", status));
  }
  buffer(buffer const &other) : memory_object(other) {}
  buffer &operator=(buffer const &other)
  { if (this != &other) memory_object::operator=(other);}
};

} // namespace ovxx::opencl
} // namespace ovxx

#endif

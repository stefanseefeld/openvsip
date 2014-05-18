//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/opencl/command_queue.hpp>
#include <ovxx/opencl/context.hpp>

namespace ovxx
{
namespace opencl
{
command_queue default_queue()
{
  static command_queue queue;
  if (!queue.get())
  {
    context c = default_context();
    device d = default_device();
    queue = command_queue(c, d);
  }
  return queue;
}

} // namespace ovxx::opencl
} // namespace ovxx

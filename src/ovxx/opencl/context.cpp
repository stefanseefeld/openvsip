//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/opencl/context.hpp>
#include <ovxx/opencl/platform.hpp>

namespace ovxx
{
namespace opencl
{
namespace
{
  context context_;
}

context default_context()
{
  if (!context_.get())
  {
    platform pl = default_platform();
    intptr_t props[] = { CL_CONTEXT_PLATFORM, (intptr_t)pl.id(), 0};
    context_ = context(device::default_, props);
  }
  return context_;
}
} // namespace ovxx::opencl
} // namespace ovxx


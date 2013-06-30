//
// Copyright (c) 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/allocator.hpp>
#include <ovxx/aligned_allocator.hpp>
#include <limits>
#include <cstdlib>

namespace ovxx
{

#if OVXX_ENABLE_THREADING
thread_local allocator *allocator::default_ = 0;
#else
allocator *allocator::default_ = 0;
#endif

void allocator::initialize(int &/*argc*/, char **&/*argv*/)
{
  default_ = new aligned_allocator();
}

void allocator::finalize()
{
  delete default_;
  default_ = 0;
}

} // namespace ovxx

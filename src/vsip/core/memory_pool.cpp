//
// Copyright (c) 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <limits>
#include <cstdlib>

#include <vsip/core/memory_pool.hpp>
#include <vsip/core/aligned_pool.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl 
{

Memory_pool::~Memory_pool()
{}

Memory_pool* default_pool = 0;

void initialize_default_pool(int& /*argc*/, char**& /*argv*/)
{
  default_pool = new Aligned_pool();
}

void finalize_default_pool()
{
  delete default_pool;
}


} // namespace vsip::impl

} // namespace vsip

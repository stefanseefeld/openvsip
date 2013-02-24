/* Copyright (c) 2007, 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/memory_pool.cpp
    @author  Jules Bergmann
    @date    2007-04-11
    @brief   VSIPL++ Library: Memory allocation pool
*/

/***********************************************************************
  Included Files
***********************************************************************/

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

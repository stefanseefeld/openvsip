/* Copyright (c) 2007, 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/aligned_pool.cpp
    @author  Jules Bergmann
    @date    2007-04-12
    @brief   VSIPL++ Library: Aligned memory allocation pool
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/config.hpp>
#include <vsip/core/memory_pool.hpp>
#include <vsip/core/aligned_pool.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl 
{

Aligned_pool::Aligned_pool()
{}

Aligned_pool::~Aligned_pool()
{}


void*
Aligned_pool::impl_allocate(size_t size)
{
  // If size == 0, allocate 1 byte.
  if (size == 0)
    size = 1;
  
  void* ptr = (void*)alloc_align<char>(align, size);
  if (ptr == 0)
    VSIP_IMPL_THROW(std::bad_alloc());
  return ptr;
}

void
Aligned_pool::impl_deallocate(void* ptr, size_t /*size*/)
{
  free_align((char*)ptr);
}

char const*
Aligned_pool::name()
{
  return "Aligned_pool";
}
  
  

} // namespace vsip::impl

} // namespace vsip

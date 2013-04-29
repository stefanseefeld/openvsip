//
// Copyright (c) 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

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

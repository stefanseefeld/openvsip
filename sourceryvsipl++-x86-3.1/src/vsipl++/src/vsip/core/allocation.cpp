/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/allocation.cpp
    @author  Jules Bergmann
    @date    2005-05-24
    @brief   VSIPL++ Library: Memory allocation functions.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cstdlib>
#include <cassert>
#include <vsip/core/allocation.hpp>



/***********************************************************************
  Included Files
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace alloc
{

/// Allocate SIZE bytes of memory with a starting address a multiple
/// of ALIGN bytes.

/// Works by allocating ALIGN extra bytes to align returned memory.
/// Stores pointer to original memory in space before aligned memory
///
/// Fallback routine for use by alloc_align if neither posix_memalign
/// nor memalign is available.


void*
impl_alloc_align(size_t align, size_t size)
{
  assert(sizeof(void*) <= align);
  assert(sizeof(void*) == sizeof(size_t));

  void*  ptr  = malloc(size + align);
  if (ptr == 0) return 0;
  size_t mask = ~(align-1);
  void*  ret  = (void*)(((size_t)ptr + align) & mask);
  *((void**)ret - 1) = ptr;
  return ret;
}



/// Free memory allocated by impl_alloc_align

void
impl_free_align(void* ptr)
{
  if (ptr) free(*((void**)ptr-1));
}



} // namespace vsip::impl::alloc

} // namespace vsip::impl

} // namespace vsip

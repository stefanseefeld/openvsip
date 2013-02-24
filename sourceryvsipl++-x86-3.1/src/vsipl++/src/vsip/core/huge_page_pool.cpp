/* Copyright (c) 2007 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/huge_page_pool.cpp
    @author  Jules Bergmann
    @date    2007-04-12
    @brief   VSIPL++ Library: Memory allocation pool from huge pages
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <limits>
#include <cstdlib>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>

#include <vsip/core/config.hpp>
#include <vsip/core/allocation.hpp>
#include <vsip/core/huge_page_pool.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

#if VSIP_IMPL_ENABLE_HUGE_PAGE_POOL

namespace vsip
{

namespace impl 
{

// Allocate memory in huge page space (that are freed on program
// termination)
//
// Requires
//   MEM_FILE to be a filename in the /huge pages directory.
//   PAGES to be the number of pages.
//
// Returns a pointer to the start of the memory if successful,
//   NULL otherwise.

char*
open_huge_pages(char const* mem_file, int pages)
{
  int   fmem;
  char* mem_addr;

  if ((fmem = open(mem_file, O_CREAT | O_RDWR, 0755)) == -1)
  {
    std::cerr << "WARNING: unable to open file " << mem_file
	      << " (errno=" << errno << " " << strerror(errno) << ")\n";
    return 0;
  }

  // Delete file so that huge pages will get freed on program termination.
  remove(mem_file);
	
  mem_addr = (char *)mmap(0, pages * 0x1000000,
			   PROT_READ | PROT_WRITE, MAP_SHARED, fmem, 0);

  if (mem_addr == MAP_FAILED)
  {
    std::cerr << "ERROR: unable to mmap file " << mem_file
	      << " (errno=" << errno << " " << strerror(errno) << ")\n";
    close(fmem);
    return 0;
  }

    // Touch each of the large pages.
    for (int i=0; i<pages; ++i)
      mem_addr[i*0x1000000 + 0x0800000] = (char) 0;

  return mem_addr;
}



/// Aligned_pool implementation.

Huge_page_pool::Huge_page_pool(const char* file, int pages)
  : pool_       (open_huge_pages(file, pages)),
    size_       (pages * 0x1000000),
    free_       (pool_),
    total_avail_(size_)
{
  *(char**)free_ = 0; // next block
  ((size_t*)free_)[1] = size_;
}

Huge_page_pool::~Huge_page_pool()
{}

void* 
Huge_page_pool::impl_allocate(size_t size)
{
  // If size == 0, allocate 1 byte.
  if (size < 2*sizeof(char*))
    size = 2*sizeof(char*);

  // Maintain 128 B alignment
  if (size % 128 != 0)
    size += 128 - (size % 128);

  char*  prev  = 0;
  char*  ptr   = free_;
  size_t avail = ptr ? ((size_t*)ptr)[1] : 0;

  while (ptr && avail < size)
  {
    prev  = ptr;
    ptr   = *(char**)ptr;
    avail = ptr ? ((size_t*)ptr)[1] : 0;
  }

  if (ptr == 0)
    VSIP_IMPL_THROW(std::bad_alloc());

  total_avail_ -= size;

  if (avail == size)
  {
    // Exact match.
    if (prev == 0)
      free_ = *(char**)ptr;
    else
      *(char**)prev = *(char**)ptr;
  }
  else
  {
    // Larger match, carve out chunk.
    if (prev == 0)
    {
      free_ = ptr + size;
    }
    else
    {
      *(char**)prev = ptr + size;
    }

    *(char**)(ptr + size) = *(char**)ptr;
    ((size_t*)(ptr + size))[1] = avail - size;
  }

  return (void*)ptr;
}

void 
Huge_page_pool::impl_deallocate(void* return_ptr, size_t size)
{
  if (size < 2*sizeof(char*))
    size = 2*sizeof(char*);

  // Maintain 128 B alignment
  if (size % 128 != 0)
    size += 128 - (size % 128);

  char*  prev  = 0;
  char*  ptr   = free_;

  while (ptr && ptr < return_ptr)
  {
    prev  = ptr;
    ptr   = *(char**)ptr;
  }

  if (ptr == 0)
  {
    // Free list empty.
    ((size_t*)(return_ptr))[1] = size;
    free_ = (char*)return_ptr;
  }
  else if (prev == 0)
  {
    assert(free_ == ptr);
    assert(ptr-(char*)return_ptr >= (ptrdiff_t)size);
    if ((ptrdiff_t)size == ptr - (char*)return_ptr)
    {
      // Insert at front of free list, merge with next entry.
      *(char**)(return_ptr)      = *(char**)ptr;
      ((size_t*)(return_ptr))[1] = size + ((size_t*)(ptr))[1];
      free_ = (char*)return_ptr;
    }
    else
    {
      // Insert at front of free list, no merge.
      *(char**)(return_ptr)      = ptr;
      ((size_t*)(return_ptr))[1] = size;
      free_ = (char*)return_ptr;
    }
  }
  else
  {
    assert(ptr-(char*)return_ptr >= (ptrdiff_t)size);
    if ((ptrdiff_t)size == ptr - (char*)return_ptr)
    {
      // Insert in middle of free list, merge
      *(char**)(return_ptr)      = *(char**)ptr;
      ((size_t*)(return_ptr))[1] = size + ((size_t*)(ptr))[1];
    }
    else
    {
      // Insert in middle of free list, no merge
      *(char**)(return_ptr)      = ptr;
      ((size_t*)(return_ptr))[1] = size;
    }

    size_t prev_size = ((size_t*)prev)[1];

    if ((ptrdiff_t)prev_size == (char*)return_ptr - prev)
    {
      // Merge with prev.
      *(char**)(prev) = *(char**)return_ptr;
      ((size_t*)(prev))[1] = size + ((size_t*)(return_ptr))[1];
    }
    else
      // No merge with prev.
      *(char**)(prev) = (char*)return_ptr;
  }

  total_avail_ += size;
}

char const* 
Huge_page_pool::name()
{
  return "Huge_page_pool";
}

} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_IMPL_ENABLE_HUGE_PAGE_POOL

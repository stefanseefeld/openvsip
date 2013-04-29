//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <limits>
#include <cstdlib>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include <ovxx/config.hpp>
#include <ovxx/huge_page_allocator.hpp>

namespace ovxx
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

char *open_huge_pages(char const* mem_file, int pages)
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

huge_page_allocator::huge_page_allocator(char const*file, int pages)
  : pool_(open_huge_pages(file, pages)),
    size_(pages * 0x1000000),
    free_(pool_),
    total_avail_(size_)
{
  *(char**)free_ = 0; // next block
  ((std::size_t*)free_)[1] = size_;
}

void *huge_page_allocator::allocate(std::size_t size)
{
  // If size == 0, allocate 1 byte.
  if (size < 2*sizeof(char*))
    size = 2*sizeof(char*);

  // Maintain 128 B alignment
  if (size % 128 != 0)
    size += 128 - (size % 128);

  char*  prev  = 0;
  char*  ptr   = free_;
  std::size_t avail = ptr ? ((std::size_t*)ptr)[1] : 0;

  while (ptr && avail < size)
  {
    prev  = ptr;
    ptr   = *(char**)ptr;
    avail = ptr ? ((std::size_t*)ptr)[1] : 0;
  }

  if (ptr == 0)
    OVXX_THROW(std::bad_alloc());

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
    ((std::size_t*)(ptr + size))[1] = avail - size;
  }

  return (void*)ptr;
}

void huge_page_allocator::deallocate(void *return_ptr, std::size_t size)
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
    ((std::size_t*)(return_ptr))[1] = size;
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
      ((std::size_t*)(return_ptr))[1] = size + ((std::size_t*)(ptr))[1];
      free_ = (char*)return_ptr;
    }
    else
    {
      // Insert at front of free list, no merge.
      *(char**)(return_ptr)      = ptr;
      ((std::size_t*)(return_ptr))[1] = size;
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
      ((std::size_t*)(return_ptr))[1] = size + ((std::size_t*)(ptr))[1];
    }
    else
    {
      // Insert in middle of free list, no merge
      *(char**)(return_ptr)      = ptr;
      ((std::size_t*)(return_ptr))[1] = size;
    }

    std::size_t prev_size = ((std::size_t*)prev)[1];

    if ((ptrdiff_t)prev_size == (char*)return_ptr - prev)
    {
      // Merge with prev.
      *(char**)(prev) = *(char**)return_ptr;
      ((std::size_t*)(prev))[1] = size + ((std::size_t*)(return_ptr))[1];
    }
    else
      // No merge with prev.
      *(char**)(prev) = (char*)return_ptr;
  }

  total_avail_ += size;
}

} // namespace ovxx

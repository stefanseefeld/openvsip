//
// Copyright (c) 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_aligned_allocator_hpp_
#define ovxx_aligned_allocator_hpp_

#include <ovxx/config.hpp>
#include <ovxx/allocator.hpp>
#include <ovxx/aligned_array.hpp>
#include <ovxx/support.hpp>

namespace ovxx
{

class aligned_allocator : public allocator
{
public:
  static size_t const align = OVXX_ALLOC_ALIGNMENT;

private:
  void *allocate(size_t size)
  {
    // If size == 0, allocate 1 byte.
    if (size == 0) size = 1;
    void* ptr = (void*)alloc_align<char>(align, size);
    if (ptr == 0) OVXX_DO_THROW(std::bad_alloc());
    return ptr;
  }
  void deallocate(void *ptr, size_t size)
  {
    free_align((char*)ptr);
  }
};
  
} // namespace ovxx

#endif

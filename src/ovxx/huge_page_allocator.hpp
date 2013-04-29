//
// Copyright (c) 2007 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_huge_page_allocator_hpp_
#define ovxx_huge_page_allocator_hpp_

#include <ovxx/allocator.hpp>
#include <ovxx/aligned_allocator.hpp>
#include <limits>
#include <cstdlib>

namespace ovxx
{

#if OVXX_ENABLE_HUGE_PAGE_ALLOCATOR
class huge_page_allocator : public allocator
{
public:
  static size_t const align = 128;

  huge_page_allocator(char const *file, int pages);

  size_t total_avail() { return total_avail_;}

private:
  void *allocate(size_t size);
  void deallocate(void *ptr, size_t size);

  char *pool_;
  size_t size_;
  char *free_;

  size_t total_avail_;
};
#else
typedef aligned_allocator huge_page_allocator;
#endif

} // namespace ovxx

#endif

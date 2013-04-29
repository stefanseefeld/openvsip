//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_HUGE_PAGE_POOL_HPP
#define VSIP_CORE_HUGE_PAGE_POOL_HPP

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

#if VSIP_IMPL_ENABLE_HUGE_PAGE_POOL
class Huge_page_pool : public Memory_pool
{
public:
  static size_t const align = 128;

  // Constructors and destructor.
public:
  Huge_page_pool(const char* file, int pages);
  ~Huge_page_pool();

  // Memory_pool accessors.
public:
  void* impl_allocate(size_t size);
  void  impl_deallocate(void* ptr, size_t size);

  char const* name();

  // Impl accessors.
public:
  size_t total_avail() { return total_avail_; }

  // Member data.
private:
  char*  pool_;
  size_t size_;
  char*  free_;

  size_t total_avail_;
};
#else
typedef Aligned_pool Huge_page_pool;
#endif



} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_CORE_HUGE_PAGE_POOL_HPP

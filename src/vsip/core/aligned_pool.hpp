//
// Copyright (c) 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_ALIGNED_POOL_HPP
#define VSIP_CORE_ALIGNED_POOL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>
#include <vsip/core/memory_pool.hpp>
#include <vsip/core/allocation.hpp>
#include <vsip/support.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl 
{

/// Aligned_pool implementation.

class Aligned_pool
  : public Memory_pool
{
public:
  static size_t const align = VSIP_IMPL_ALLOC_ALIGNMENT;

  // Constructors and destructor.
public:
  Aligned_pool();
  ~Aligned_pool();

  // Accessors.
public:
  void* impl_allocate(size_t size);
  void  impl_deallocate(void* ptr, size_t size);

  char const* name();
};
  
  

} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_CORE_POOL_HPP

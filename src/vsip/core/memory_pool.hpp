//
// Copyright (c) 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_MEMORY_POOL_HPP
#define VSIP_CORE_MEMORY_POOL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <limits>
#include <cstdlib>
#include <stdexcept>
#include <vsip/support.hpp>
#include <vsip/core/profile.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl 
{

/// Memory_pool base class.

class Memory_pool
{
public:
  Memory_pool() {}
  virtual ~Memory_pool();

  virtual void* impl_allocate(size_t size) = 0;
  virtual void  impl_deallocate(void* ptr, size_t size) = 0;

  virtual char const* name() = 0;

  // Convenience functions
  template <typename T>
  T* allocate(length_type size)
  {
    using namespace vsip::impl::profile;
    event<memory>("Memory_pool::allocate", size * sizeof(T));
    return (T*)(impl_allocate(size * sizeof(T)));
  }

  template <typename T>
  void deallocate(T *ptr, length_type size)
  {
    using namespace vsip::impl::profile;
    event<memory>("Memory_pool::deallocate", size * sizeof(T));
    impl_deallocate(ptr, size * sizeof(T));
  }
};


extern Memory_pool* default_pool;

void initialize_default_pool(int& argc, char**&argv);

void finalize_default_pool();


} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_CORE_POOL_HPP

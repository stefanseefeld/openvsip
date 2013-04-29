/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/aligned_allocator.hpp
    @author  Jules Bergmann
    @date    2005-05-23
    @brief   VSIPL++ Library: Aligned Allocator

    Based on default allocator from Josuttis Ch. 15.
*/

#ifndef VSIP_CORE_ALIGNED_ALLOCATOR_HPP
#define VSIP_CORE_ALIGNED_ALLOCATOR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <limits>
#include <cstdlib>

#include <vsip/core/config.hpp>
#include <vsip/core/allocation.hpp>
#include <vsip/core/profile.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl 
{


/// Allocator for aligned data.

template <typename T>
class Aligned_allocator
{
  // Type definitions.
public:
  typedef T              value_type;
  typedef T*             pointer;
  typedef T const*       const_pointer;
  typedef T&             reference;
  typedef T const&       const_reference;
  typedef std::size_t    size_type;
  typedef std::ptrdiff_t difference_type;

  // Constants.
public:
  // Alignment (in bytes)
  static size_t const align = VSIP_IMPL_ALLOC_ALIGNMENT;

  // rebind allocator to type U
  template <class U>
  struct rebind
  {
    typedef Aligned_allocator<U> other;
  };

  // Constructors and destructor.
public:
  Aligned_allocator() throw() {}
  Aligned_allocator(Aligned_allocator const&) throw() {}

  template <class U>
  Aligned_allocator (Aligned_allocator<U> const&) throw() {}

  ~Aligned_allocator() throw() {}
  
  
  // return address of values
  pointer address (reference value) const
    { return &value; }

  const_pointer address (const_reference value) const
    { return &value; }
  
  
  // return maximum number of elements that can be allocated
  size_type max_size() const throw()
    { return std::numeric_limits<std::size_t>::max() / sizeof(T); }
  
  // allocate but don't initialize num elements of type T
  pointer allocate(size_type num, const void* = 0)
  {
    using namespace vsip::impl::profile;
    event<memory>("Aligned_allocator::allocate",
		  num * sizeof(value_type));    
    // If num == 0, allocate 1 element.
    if (num == 0)
      num = 1;
    
    // allocate aligned memory
    pointer p = alloc_align<value_type>(align, num);
    if (p == 0)
    {
      printf("failed to allocate(%lu)\n", static_cast<unsigned long>(num));
      VSIP_IMPL_THROW(std::bad_alloc());
    }
    return p;
  }
  
  // initialize elements of allocated storage p with value value
  void construct(pointer p, const T& value)
  {
    // initialize memory with placement new
    new((void*)p)T(value);
  }
  
  // destroy elements of initialized storage p
  void destroy(pointer p)
  {
    // destroy objects by calling their destructor
    p->~T();
  }
  
  // deallocate storage p of deleted elements
  void deallocate(pointer p, size_type num)
  {
    using namespace vsip::impl::profile;
    event<memory>("Aligned_allocator::deallocate",
		  num * sizeof(value_type));
    free_align(p);
  }
};



/***********************************************************************
  Definitions
***********************************************************************/

// return that all specializations of this allocator are interchangeable
template <typename T1,
	  typename T2>
bool operator== (Aligned_allocator<T1> const&,
		 Aligned_allocator<T2> const&) throw()
{
  return true;
}

template <typename T1,
	  typename T2>
bool operator!= (Aligned_allocator<T1> const&,
		 Aligned_allocator<T2> const&) throw()
{
  return false;
}

} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_CORE_ALIGNED_ALLOCATOR_HPP

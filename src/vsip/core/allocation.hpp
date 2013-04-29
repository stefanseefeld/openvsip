//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_ALLOCATION_HPP
#define VSIP_CORE_ALLOCATION_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/config.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/profile.hpp>
#include <complex>
#include <cstdlib>
#include <cstring>

#if defined(HAVE_MALLOC_H)
#  include <malloc.h>
#endif




/***********************************************************************
  Declarations and definitions.
***********************************************************************/

#if defined(HAVE_DECL_POSIX_MEMALIGN) && !HAVE_DECL_POSIX_MEMALIGN
extern "C" {
extern int posix_memalign(void**, size_t, size_t);
}
#endif

#if defined(HAVE_DECL_MEMALIGN) && !HAVE_DECL_MEMALIGN
extern "C" {
extern void* memalign(size_t, size_t);
}
#endif



namespace vsip
{

namespace impl 
{

namespace alloc
{

/// Internal routine to allocate aligned memory, used by alloc_align.
/// Elsewhere in the library, alloc_align or Aligned_allocator should
/// be used.

void*
impl_alloc_align(size_t align, size_t size);



/// Internal routine to free memory allocated with impl_alloc_align.
/// Used by free_align.

void
impl_free_align(void* ptr);

} // namespace vsip::impl::alloc



/// Allocate aligned memory.

template <typename T>
inline T*
alloc_align(size_t align, size_t size)
{
  void* ptr;
#if HAVE_POSIX_MEMALIGN && !VSIP_IMPL_AVOID_POSIX_MEMALIGN
  if (posix_memalign(&ptr, align, size*sizeof(T)) != 0)
    throw std::bad_alloc();
#elif HAVE_MEMALIGN
  ptr = memalign(align, size*sizeof(T));
  if (!ptr) throw std::bad_alloc();
#else
  ptr = alloc::impl_alloc_align(align, size*sizeof(T));
  if (!ptr) throw std::bad_alloc();
#endif
  return static_cast<T*>(ptr);
}



/// Free aligned memory.

inline void
free_align(void* ptr)
{
#if ((HAVE_POSIX_MEMALIGN && !VSIP_IMPL_AVOID_POSIX_MEMALIGN) || HAVE_MEMALIGN)
  free(ptr);
#else
  alloc::impl_free_align(ptr);
#endif
}

template <typename T>
struct aligned_array_ref
{
  explicit aligned_array_ref(T *p, size_t s) : ptr(p), size(s) {}
  T *ptr;
  size_t size;
};

template <typename T>
class aligned_array
{
public:
  typedef T value_type;

  explicit 
  aligned_array(size_t size)
    : size_(size), data_(alloc_align<T>(VSIP_IMPL_ALLOC_ALIGNMENT, size))
  {
    namespace p = vsip::impl::profile;
    p::event<p::memory>("aligned_array(size_t)", size_ * sizeof(T));
  }
  explicit 
  aligned_array(size_t alignment, size_t size, T *data = 0)
    : size_(size), data_(alloc_align<T>(alignment, size))
  {
    namespace p = vsip::impl::profile;
    p::event<p::memory>("aligned_array(size_t, size_t, T *)", size_ * sizeof(T));
    if (data) memcpy(data_, data, size * sizeof(T));
  }
  aligned_array(aligned_array &a) : size_(a.size()), data_(a.get()) { a.release();}
  aligned_array(aligned_array_ref<value_type> r) : size_(r.size), data_(r.ptr) {}
  ~aligned_array()
  {
    namespace p = vsip::impl::profile;
    p::event<p::memory>("~aligned_array()", size_ * sizeof(T));
    free_align(data_);
  }
  aligned_array &operator= (aligned_array &a)
  {
    size_t s = a.size();
    reset(a.release());
    size_ = s;
    return *this;
  }
  aligned_array &operator=(aligned_array_ref<value_type> r)
  {
    reset(r.ptr, r.size);
    return *this;
  }
  size_t size() const { return size_;}

  value_type& operator[](size_t i) { return data_[i];}
  value_type* get() const throw() { return data_;}

  template<typename T1>
  operator aligned_array_ref<T1>()
  {
    size_t s = size_;
    return aligned_array_ref<T1>(this->release(), s);
  }

private:
  value_type *release()
  {
    value_type *tmp = data_;
    data_ = 0;
    size_ = 0;
    return tmp;
  }
  void reset(value_type *p = 0, size_t s = 0)
  {
    if (p != data_)
    {
      using namespace vsip::impl::profile;
      event<memory>("aligned_array::reset()", size_ * sizeof(T));
      free_align(data_);
      data_ = p;
      size_ = s;
    }
  }

  size_t size_;
  T *data_;
};

template <typename T, storage_format_type C>
struct Complex_format_cast
{
  typedef T *type;
  static type
  cast(aligned_array<T> &a) { return a.get();}
};
template <typename T>
struct Complex_format_cast<std::complex<T>, interleaved_complex>
{
  typedef std::complex<T> *type;
  static type
  cast(aligned_array<std::complex<T> > &a) { return a.get();}
};
template <typename T>
struct Complex_format_cast<std::complex<T>, split_complex>
{
  typedef std::pair<T*, T*> type;
  static type
  cast(aligned_array<std::complex<T> > &a) 
  {
    T *ptr = reinterpret_cast<T*>(a.get());
    return std::make_pair(ptr, ptr + a.size());
  }
};

template <storage_format_type C, typename T> 
typename Complex_format_cast<T, C>::type
array_cast(aligned_array<T> &a)
{ return Complex_format_cast<T, C>::cast(a);}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_ALLOCATION_HPP

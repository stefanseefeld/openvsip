//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_aligned_array_hpp_
#define ovxx_aligned_array_hpp_

#include <ovxx/config.hpp>
#include <ovxx/layout.hpp>
#include <ovxx/complex_decl.hpp>
#include <cstdlib>
#include <cstring>
#include <cassert>

#if defined(HAVE_MALLOC_H)
#  include <malloc.h>
#endif

#if defined(HAVE_DECL_POSIX_MEMALIGN) && !HAVE_DECL_POSIX_MEMALIGN
extern "C" extern int posix_memalign(void**, std::size_t, std::size_t);
#endif

#if defined(HAVE_DECL_MEMALIGN) && !HAVE_DECL_MEMALIGN
extern "C" extern void* memalign(std::size_t, std::size_t);
#endif

namespace ovxx
{
namespace detail
{

/// Internal routine to allocate aligned memory, used by alloc_align.
/// Elsewhere in the library, alloc_align or Aligned_allocator should
/// be used.
inline void *alloc_align(std::size_t align, std::size_t size)
{
  OVXX_PRECONDITION(sizeof(void*) <= align);
  OVXX_PRECONDITION(sizeof(void*) == sizeof(std::size_t));

  void*  ptr  = malloc(size + align);
  if (ptr == 0) return 0;
  std::size_t mask = ~(align-1);
  void*  ret  = (void*)(((std::size_t)ptr + align) & mask);
  *((void**)ret - 1) = ptr;
  return ret;
}

inline void free_align(void *ptr)
{
  if (ptr) free(*((void**)ptr-1));
}

} // namespace ovxx::detail

template <typename T>
inline T *alloc_align(std::size_t align, std::size_t size)
{
  void *ptr;
#if HAVE_POSIX_MEMALIGN
  if (posix_memalign(&ptr, align, size*sizeof(T)) != 0)
    throw std::bad_alloc();
#elif HAVE_MEMALIGN
  ptr = memalign(align, size*sizeof(T));
  if (!ptr) throw std::bad_alloc();
#else
  ptr = detail::alloc_align(align, size*sizeof(T));
  if (!ptr) throw std::bad_alloc();
#endif
  return static_cast<T*>(ptr);
}

inline void free_align(void *ptr)
{
#if (HAVE_POSIX_MEMALIGN || HAVE_MEMALIGN)
  free(ptr);
#else
  detail::free_align(ptr);
#endif
}

template <typename T>
struct aligned_array_ref
{
  explicit aligned_array_ref(T *p, std::size_t s) : ptr(p), size(s) {}
  T *ptr;
  std::size_t size;
};

template <typename T>
class aligned_array
{
public:
  typedef T value_type;

  aligned_array() : size_(0), data_(0) {}

  explicit 
  aligned_array(std::size_t size)
    : size_(size), data_(alloc_align<T>(OVXX_ALLOC_ALIGNMENT, size))
  {
  }
  explicit 
  aligned_array(std::size_t alignment, std::size_t size, T const *data = 0)
    : size_(size), data_(alloc_align<T>(alignment, size))
  {
    if (data) memcpy(data_, data, size * sizeof(T));
  }
  aligned_array(aligned_array &a) : size_(a.size()), data_(a.get()) { a.release();}
  aligned_array(aligned_array_ref<value_type> r) : size_(r.size), data_(r.ptr) {}
  ~aligned_array() { free_align(data_);}
  aligned_array &operator= (aligned_array &a)
  {
    std::size_t s = a.size();
    reset(a.release());
    size_ = s;
    return *this;
  }
  aligned_array &operator=(aligned_array_ref<value_type> r)
  {
    reset(r.ptr, r.size);
    return *this;
  }
  std::size_t size() const { return size_;}

  value_type &operator[](std::size_t i) { return data_[i];}
  value_type const &operator[](std::size_t i) const { return data_[i];}
  value_type *get() throw() { return data_;}
  value_type const *get() const throw() { return data_;}

  template<typename T1>
  operator aligned_array_ref<T1>()
  {
    std::size_t s = size_;
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
  void reset(value_type *p = 0, std::size_t s = 0)
  {
    if (p != data_)
    {
      free_align(data_);
      data_ = p;
      size_ = s;
    }
  }

  std::size_t size_;
  T *data_;
};

namespace detail
{
template <typename T, storage_format_type S>
struct array_cast
{
  typedef T *type;
  static type
  cast(aligned_array<T> &a) { return a.get();}
};
template <typename T>
struct array_cast<complex<T>, interleaved_complex>
{
  typedef complex<T> *type;
  static type
  cast(aligned_array<complex<T> > &a) { return a.get();}
};
template <typename T>
struct array_cast<complex<T>, split_complex>
{
  typedef std::pair<T*, T*> type;
  static type
  cast(aligned_array<complex<T> > &a) 
  {
    T *ptr = reinterpret_cast<T*>(a.get());
    return std::make_pair(ptr, ptr + a.size());
  }
};

} // namespace ovxx::detail

template <storage_format_type S, typename T> 
typename detail::array_cast<T, S>::type
array_cast(aligned_array<T> &a)
{ return detail::array_cast<T, S>::cast(a);}

} // namespace ovxx

#endif

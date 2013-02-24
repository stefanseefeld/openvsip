/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_CORE_DENSE_STORAGE_HPP
#define VSIP_CORE_DENSE_STORAGE_HPP

#include <vsip/support.hpp>
#include <vsip/core/storage.hpp>
#include <vsip/core/memory_pool.hpp>

namespace vsip
{
namespace impl
{ 

template <storage_format_type C, typename T>
class Dense_storage
{
public:
  typedef T*       type;
  typedef T const* const_type;

  Dense_storage(Memory_pool*  pool,
		length_type   size,
		type          buffer = 0)
    VSIP_THROW((std::bad_alloc))
    : alloc_data_(!buffer),
      data_      (alloc_data_ ? pool->allocate<T>(size) : (T*)buffer)
  {}

  Dense_storage(Memory_pool*  pool,
		length_type   size,
		T             val,
		type          buffer = 0)
  VSIP_THROW((std::bad_alloc))
    : alloc_data_(!buffer),
      data_      (alloc_data_ ? pool->allocate<T>(size) : (T*)buffer)
  {
    for (index_type i=0; i<size; ++i)
      data_[i] = val;
  }

  ~Dense_storage()
  {
    // user's responsiblity to call deallocate().
    if (alloc_data_)
      assert(data_ == 0);
  }

  T    get(index_type idx) const { return data_[idx];}
  void put(index_type idx, T val){ data_[idx] = val;}

  T &ref(index_type idx) { return data_[idx];}
  T const &ref(index_type idx) const { return data_[idx];}

  type ptr() { return data_;}
  const_type ptr() const { return data_;}

  /// Rebind the memory referred to by Dense_storage object
  ///
  /// Requires:
  ///   :size: size object was constructed with.
  void rebind(Memory_pool* pool, length_type size, type buffer)
  {
    if (buffer)
    {
      if (alloc_data_) pool->deallocate<T>(data_, size);
      alloc_data_ = false;
      data_       = buffer;
    }
    else // (buffer == 0
    {
      if (!alloc_data_)
      {
	alloc_data_ = true;
	data_ = pool->allocate<T>(size);
      }
      /* else do nothing - we already own our data */
    }
  }

  void deallocate(Memory_pool* pool, length_type size)
  {
    if (alloc_data_)
    {
      pool->deallocate(data_, size);
      data_ = 0;
      alloc_data_ = false;
    }
  }

  bool is_alloc() const { return alloc_data_; }

private:
  bool   alloc_data_;
  T*     data_;
};

template <typename T>
class Dense_storage<split_complex, complex<T> >
{
public:
  typedef std::pair<T*, T*>             type;
  typedef std::pair<T const*, T const*> const_type;

  Dense_storage(Memory_pool*  pool,
		length_type   size,
		type          buffer    = type(0, 0))
    VSIP_THROW((std::bad_alloc))
    : alloc_data_(!buffer.first || !buffer.second),
      real_data_ (alloc_data_ ? pool->allocate<T>(size) : buffer.first),
      imag_data_ (alloc_data_ ? pool->allocate<T>(size) : buffer.second)
  {}

  Dense_storage(Memory_pool*     pool,
		length_type      size,
		vsip::complex<T> val,
		type buffer = type(0, 0))
    VSIP_THROW((std::bad_alloc))
    : alloc_data_(!buffer.first || !buffer.second),
      real_data_ (alloc_data_ ? pool->allocate<T>(size) : buffer.first),
      imag_data_ (alloc_data_ ? pool->allocate<T>(size) : buffer.second)
  {
    for (index_type i=0; i<size; ++i)
      real_data_[i] = val.real();
    for (index_type i=0; i<size; ++i)
      imag_data_[i] = val.imag();
  }

  ~Dense_storage()
  {
    // user's responsiblity to call deallocate().
    if (alloc_data_)
      assert(real_data_ == 0 && imag_data_ == 0);
  }

  vsip::complex<T> get(index_type idx) const
  { return vsip::complex<T>(real_data_[idx], imag_data_[idx]);}

  void put(index_type idx, vsip::complex<T> val)
  {
    real_data_[idx] = val.real();
    imag_data_[idx] = val.imag();
  }

  vsip::complex<T> &ref(index_type)
  {
    VSIP_IMPL_THROW(unimplemented(
	"Dense_storage<split_complex>::ref - unimplemented"));
  }
  vsip::complex<T> const &ref(index_type) const
  {
    VSIP_IMPL_THROW(unimplemented(
	"Dense_storage<split_complex>::ref - unimplemented"));
  }

  type ptr() { return type(real_data_, imag_data_);}
  const_type ptr() const { return const_type(real_data_, imag_data_);}

  /// Rebind the memory referred to by Dense_storage object
  ///
  /// Requires:
  ///   :size: size object was constructed with.
  void rebind(Memory_pool* pool, length_type size, type buffer)
  {
    if (buffer.first && buffer.second)
    {
      if (alloc_data_)
      {
	pool->deallocate(real_data_, size);
	pool->deallocate(imag_data_, size);
      }
      alloc_data_ = false;
      real_data_  = buffer.first;
      imag_data_  = buffer.second;
    }
    else
    {
      if (!alloc_data_)
      {
	alloc_data_ = true;
	real_data_ = pool->allocate<T>(size);
	imag_data_ = pool->allocate<T>(size);
      }
      /* else do nothing - we already own our data */
    }
  }

  void deallocate(Memory_pool* pool, length_type size)
  {
    if (alloc_data_)
    {
      pool->deallocate(real_data_, size);
      pool->deallocate(imag_data_, size);
      real_data_ = 0;
      imag_data_ = 0;
      alloc_data_ = false;
    }
  }

  bool is_alloc() const { return alloc_data_;}

private:
  bool   alloc_data_;
  T*     real_data_;
  T*     imag_data_;
};

} // namespace vsip::impl
} // namespace vsip

#endif

//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_storage_hpp_
#define ovxx_opencl_storage_hpp_

#include <ovxx/storage/storage.hpp>
#include <ovxx/pointer.hpp>
#include <ovxx/opencl/buffer.hpp>
#include <ovxx/opencl/command_queue.hpp>

namespace ovxx
{
namespace opencl
{
namespace detail
{
template <typename T>
buffer allocate(length_type size)
{
  return buffer(default_context(), size*sizeof(T), buffer::read_write);
}

template <typename T>
inline void deallocate(buffer buffer, length_type size) {}

template <typename T>
void copy_to_host(T *dest, buffer src, length_type size)
{
  default_queue().read(src, dest, size);
}

template <typename T>
void copy_from_host(buffer dest, T const *src, length_type size)
{
  default_queue().write(src, dest, size);
}

} // namespace ovxx::opencl::detail


template <typename T, storage_format_type F>
class storage : ovxx::detail::noncopyable
{
  typedef storage_traits<T, F> t;
public:
  typedef typename t::value_type value_type;
  typedef typename t::ptr_type ptr_type;
  typedef typename t::const_ptr_type const_ptr_type;
  // typedef buffer ptr_type;
  // typedef buffer const_ptr_type;
  typedef typename t::reference_type reference_type;
  typedef typename t::const_reference_type const_reference_type;

  storage(length_type size, bool allocate=true)
    : size_(size), data_(), valid_(true)
  { if (allocate) this->allocate();}
  virtual ~storage() { deallocate();}
  virtual void allocate()
  {
    if (!data_.get())
      data_ = detail::allocate<T>(this->size_);
  }
  virtual void deallocate() { data_ = buffer();}
  virtual void resize(length_type size)
  {
    if (this->size_ == size) return;
    deallocate();
    this->size_ = size;
    allocate();
  }

  void invalidate() { valid_ = false;}
  void validate() { valid_ = true;}
  bool is_valid() const { return valid_;}

  buffer ptr() { return data_;}
  length_type size() const { return size_;}

  void copy_to_host(ptr_type p) const
  {
    detail::copy_to_host(p, this->data_, this->size_);
  }
  void copy_from_host(const_ptr_type p)
  {
    detail::copy_from_host(this->data_, p, this->size_);
  }
private:
  length_type size_;
  buffer data_;
  bool valid_;
};

template <typename T>
class storage<complex<T>, interleaved_complex> : ovxx::detail::noncopyable
{
  typedef storage_traits<complex<T>, interleaved_complex> t;
public:
  typedef typename t::value_type value_type;
  typedef typename t::ptr_type ptr_type;
  typedef typename t::const_ptr_type const_ptr_type;
  // typedef buffer ptr_type;
  // typedef buffer const_ptr_type;
  typedef typename t::reference_type reference_type;
  typedef typename t::const_reference_type const_reference_type;

  storage(length_type size, bool allocate=true)
    : size_(size), data_(), valid_(true)
  { if (allocate) this->allocate();}
  virtual ~storage() { this->deallocate();}
  virtual void allocate()
  {
    if (!data_.get())
      data_ = detail::allocate<T>(2*this->size_);
  }
  virtual void deallocate()
  {
    data_ = buffer();
  }
  virtual void resize(length_type size)
  {
    if (this->size_ == size) return;
    else deallocate();
    this->size_ = size;
    allocate();
  }

  void invalidate() { valid_ = false;}
  void validate() { valid_ = true;}
  bool is_valid() const { return valid_;}

  buffer ptr() { return data_;}
  length_type size() const { return size_;}

  void copy_to_host(ptr_type p) const
  {
    detail::copy_to_host(p, data_, 2*this->size_);
  }
  void copy_from_host(const_ptr_type p)
  {
    detail::copy_from_host(data_, p, 2*this->size_);
  }
private:
  length_type size_;
  buffer data_;
  bool valid_;
};

template <typename T>
class storage<complex<T>, split_complex> : ovxx::detail::noncopyable
{
  typedef storage_traits<complex<T>, split_complex> t;
public:
  typedef typename t::value_type value_type;
  typedef typename t::ptr_type ptr_type;
  typedef typename t::const_ptr_type const_ptr_type;
  // typedef std::pair<buffer,buffer> ptr_type;
  // typedef std::pair<buffer,buffer> const_ptr_type;
  typedef typename t::reference_type reference_type;
  typedef typename t::const_reference_type const_reference_type;

  storage(length_type size, bool allocate=true)
    : size_(size), data_(), valid_(true)
  { if (allocate) this->allocate();}
  virtual ~storage() { deallocate();}
  virtual void allocate()
  {
    if (!data_.first.get())
    {
      data_.first = detail::allocate<T>(this->size_);
      data_.second = detail::allocate<T>(this->size_);
    }
  }
  virtual void deallocate()
  {
    data_.first = buffer();
    data_.second = buffer();
  }
  virtual void resize(length_type size)
  {
    if (this->size_ == size) return;
    else deallocate();
    this->size_ = size;
    allocate();
  }

  void invalidate() { valid_ = false;}
  void validate() { valid_ = true;}
  bool is_valid() const { return valid_;}

  ptr_type ptr()
  { return std::make_pair(data_.first.get(), data_.second.get());}
  length_type size() const { return size_;}

  void copy_to_host(ptr_type p) const
  {
    detail::copy_to_host(p.first, data_.first, this->size_);
    detail::copy_to_host(p.second, data_.second, this->size_);
  }
  void copy_from_host(const_ptr_type p)
  {
    detail::copy_from_host(data_.first, p.first, this->size_);
    detail::copy_from_host(data_.second, p.second, this->size_);
  }
private:
  length_type size_;
  std::pair<buffer, buffer> data_;
  bool valid_;
};

template <typename T>
class storage<complex<T>, any_storage_format>
  : public ovxx::storage<complex<T>, any_storage_format>
{
public:
  typedef typename ovxx::storage<complex<T>, any_storage_format>::ptr_type ptr_type;
  typedef typename ovxx::storage<complex<T>, any_storage_format>::const_ptr_type const_ptr_type;

  storage(length_type size, storage_format_type f, bool allocate=true)
    : ovxx::storage<complex<T>, any_storage_format>(size),
      format_(f)
  { if (allocate) this->allocate();}
  virtual ~storage()
  {
    deallocate();
  }
  void allocate()
  {
    if (this->data_ != ptr_type()) return;
    switch (format_)
    {
      case array:
	this->data_ = detail::allocate<complex<T> >(this->size_);
	break;
      case interleaved_complex:
	this->data_ = detail::allocate<T>(2*this->size_);
	break;
      case split_complex:
	this->data_ = std::make_pair(detail::allocate<T>(this->size_),
				     detail::allocate<T>(this->size_));
	break;
      default: assert(0);
    }
  }
  void deallocate()
  {
    switch (format_)
    {
      case array:
      {
	complex<T> *ptr = this->data_.template as<array>();
	detail::deallocate(ptr);
	break;
      }
      case interleaved_complex:
      {
	T *ptr = this->data_.template as<interleaved_complex>();
	detail::deallocate(ptr);
	break;
      }
      case split_complex:
      {
	std::pair<T*,T*> ptr = this->data_.template as<split_complex>();
	detail::deallocate(ptr.second);
	detail::deallocate(ptr.first);
	break;
      }
      default: assert(0);
    }
    this->data_ = ptr_type();
  }
  virtual void resize(length_type size)
  {
    if (this->size_ == size) return;
    else deallocate();
    this->size_ = size;
    allocate();
  }

  storage_format_type format() const { return format_;}

  void copy_to_host(ptr_type p) const
  {
    switch (format_)
    {
      case array:
	detail::copy_to_host(p.template as<array>(),
			     this->data_.template as<array>(),
			     this->size_);
	break;
      case interleaved_complex:
	detail::copy_to_host(p.template as<interleaved_complex>(),
			     this->data_.template as<interleaved_complex>(),
			     2*this->size_);
	break;
      case split_complex:
      {
	std::pair<T*,T*> dest = p.template as<split_complex>();
	std::pair<T const*,T const*> src = this->data_.template as<split_complex>();
	detail::copy_to_host(dest.first, src.first, this->size_);
	detail::copy_to_host(dest.second, src.second, this->size_);
	break;
      }
      default:
	assert(0);
    }
  }
  void copy_from_host(const_ptr_type p)
  {
    switch (format_)
    {
      case array:
	detail::copy_from_host(this->data_.template as<array>(),
			       p.template as<array>(),
			       this->size_);
	break;
      case interleaved_complex:
	detail::copy_from_host(this->data_.template as<interleaved_complex>(),
			       p.template as<interleaved_complex>(),
			       2*this->size_);
	break;
      case split_complex:
      {
	std::pair<T*,T*> dest = this->data_.template as<split_complex>();
	std::pair<T const*,T const*> src = p.template as<split_complex>();
	detail::copy_from_host(dest.first, src.first, this->size_);
	detail::copy_from_host(dest.second, src.second, this->size_);
	break;
      }
      default:
	assert(0);
    }
  }

private:

  storage_format_type format_;
};

} // namespace ovxx::opencl
} // namespace ovxx

#endif

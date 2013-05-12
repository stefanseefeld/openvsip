//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_storage_user_hpp_
#define ovxx_storage_user_hpp_

#include <ovxx/storage/storage.hpp>

namespace ovxx
{
template <typename T> class user_storage;

template <storage_format_type F, typename T>
typename storage_traits<T, F>::ptr_type
storage_cast(user_storage<T> &s)
{ return detail::storage_cast<T, F>::exec(s);}

template <storage_format_type F, typename T>
typename storage_traits<T, F>::const_ptr_type
storage_cast(user_storage<T> const &s)
{ return detail::storage_cast<T, F>::exec(s);}

template <typename T>
class user_storage
{
public:
  user_storage()
    : format_(no_user_format), data_(0), size_(0) {}
  user_storage(T *data, length_type size)
    : format_(data ? array_format : no_user_format), data_(data), size_(size) {}

  operator bool() { return format_ != no_user_format;}
  user_storage_type format() const { return format_;}
  T const *ptr() const { return data_;}
  T *ptr() { return data_;}
  template <storage_format_type F>
  typename storage_traits<T, F>::const_ptr_type as() const { return ptr();}
  template <storage_format_type F>
  typename storage_traits<T, F>::ptr_type as() { return ptr();}
  length_type size() const { return size_;}

  void find(T *&ptr) { ptr = data_;}
  void rebind(T *ptr) { data_ = ptr;}
  void rebind(T *ptr, length_type size)
  {
    this->data_ = ptr;
    this->size_ = size;
  }

  T get(index_type i) const { return data_[i];}
  void put(index_type i, T v) { data_[i] = v;}
  T &at(index_type i) { return data_[i];}
  T const &at(index_type i) const { return data_[i];}

private:
  user_storage_type format_;
  T* data_;
  length_type size_;
};

template <typename T>
class user_storage<complex<T> >
{
  template <typename, storage_format_type>
  friend struct detail::storage_cast;
public:
  user_storage()
    : format_(no_user_format), size_(0) { data_.a = 0;}
  user_storage(complex<T> *data, length_type size)
    : format_(data ? array_format : no_user_format), size_(size)
  { data_.a = data;}
  user_storage(T *data, length_type size)
    : format_(data ? interleaved_format : no_user_format), size_(size)
  { data_.i = data;}
  user_storage(std::pair<T*,T*> data, length_type size)
    : format_(data.first ? split_format : no_user_format), size_(size)
  {
    data_.s.first=data.first;
    data_.s.second=data.second;
  }

  user_storage_type format() const { return format_;}
  template <storage_format_type F>
  typename storage_traits<complex<T>, F>::const_ptr_type as() const
  { return storage_cast<F>(*this);}
  template <storage_format_type F>
  typename storage_traits<complex<T>, F>::ptr_type as()
  { return storage_cast<F>(*this);}
  length_type size() const { return size_;}

  void find(complex<T> *&ptr) { ptr = this->data_.a;}
  void find(T *&ptr) { ptr = this->data_.i;}
  void find(std::pair<T*,T*> &ptr)
  { ptr = std::make_pair(this->data_.s.first, this->data_.s.second);}

  void rebind(complex<T> *ptr)
  {
    this->data_.a = ptr;
    this->format_ = array_format;
  }
  void rebind(complex<T> *ptr, length_type size)
  {
    this->data_.a = ptr;
    this->format_ = array_format;
    this->size_ = size;
  }
  void rebind(T *ptr)
  {
    this->data_.i = ptr;
    this->format_ = interleaved_format;
  }
  void rebind(T *ptr, length_type size)
  {
    this->data_.i = ptr;
    this->format_ = interleaved_format;
    this->size_ = size;
  }
  void rebind(std::pair<T*,T*> ptr)
  {
    this->data_.s.first = ptr.first;
    this->data_.s.second = ptr.second;
    this->format_ = split_format;
  }
  void rebind(std::pair<T*,T*> ptr, length_type size)
  {
    this->data_.s.first = ptr.first;
    this->data_.s.second = ptr.second;
    this->format_ = split_format;
    this->size_ = size;
  }

  complex<T> get(index_type i) const
  {
    OVXX_PRECONDITION(format_ != no_user_format);
    switch (format_)
    {
      case array_format: return data_.a[i];
      case interleaved_format: return complex<T>(data_.i[2*i], data_.i[2*i+1]);
      case split_format: return complex<T>(data_.s.first[i], data_.s.second[i]);
      default: assert(0);
    }
  }
  void put(index_type i, complex<T> const &v)
  {
    assert(format_ != no_user_format);
    switch (format_)
    {
      case array_format: data_.a[i] = v; break;
      case interleaved_format:
	data_.i[2*i] = v.real();
	data_.i[2*i+1] = v.imag();
	break;
      case split_format:
	data_.s.first[i] = v.real();
	data_.s.second[i] = v.imag();
	break;
      default: assert(0);
    }
  }
  complex<T> &at(index_type i)
  {
    assert(format_ == array_format || format_ == interleaved_format);
    return data_.a[i];
  }
  complex<T> const &at(index_type i) const
  {
    assert(format_ == array_format || format_ == interleaved_format);
    return data_.a[i];
  }

private:
  typename storage<complex<T>, array>::const_ptr_type
  as_array() const
  {
    assert(format_ == array_format || format_ == interleaved_format);
    return data_.a;
  }
  typename storage<complex<T>, array>::ptr_type
  as_array()
  {
    assert(format_ == array_format || format_ == interleaved_format);
    return data_.a;
  }
  typename storage<complex<T>, interleaved_complex>::const_ptr_type
  as_interleaved() const
  {
    assert(format_ == array_format || format_ == interleaved_format);
    return data_.i;
  }
  typename storage<complex<T>, interleaved_complex>::ptr_type
  as_interleaved()
  {
    assert(format_ == array_format || format_ == interleaved_format);
    return data_.i;
  }
  typename storage<complex<T>, split_complex>::const_ptr_type
  as_split() const
  {
    assert(format_ == split_format);
    return std::make_pair(data_.s.first,data_.s.second);
  }
  typename storage<complex<T>, split_complex>::ptr_type
  as_split()
  {
    assert(format_ == split_format);
    return std::make_pair(data_.s.first, data_.s.second);
  }

  user_storage_type format_;
  union
  {
    complex<T> *a;
    T *i;
    // std::pair<T*,T*> is only allowed since C++11 here.
    struct { T*first, *second;} s;
  } data_;
  length_type size_;
};

} // namespace ovxx

#endif

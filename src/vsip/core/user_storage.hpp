//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_USER_STORAGE_HPP
#define VSIP_CORE_USER_STORAGE_HPP

#include <vsip/core/storage.hpp>

namespace vsip
{

enum user_storage_type 
{
  no_user_format = 0,
  array_format,
  interleaved_format,
  split_format
};

namespace impl
{

template <typename T> class User_storage;

template <typename T, storage_format_type C>
struct User_storage_as_helper;

template <typename T>
struct User_storage_as_helper<T, interleaved_complex>
{
  static
  typename Storage<interleaved_complex, T>::type 
  exec(User_storage<T> const &s) { return s.as_interleaved();}
};

template <typename T>
struct User_storage_as_helper<T, split_complex>
{
  static
  typename Storage<split_complex, T>::type 
  exec(User_storage<T> const &s) { return s.as_split();}
};

/// Class to hold storage that has been given to us by the user.
/// For complex data, this can be in three different formats (array
/// of complex, array of interleaved real, or arrays of split real).
/// The block value type does not determine/restrict which type of
/// data the block can be constructed with.
///
/// General case intended for non-complex types.
template <typename T>
class User_storage
{
public:
  User_storage() : format_(no_user_format), data_(0) {}
  User_storage(user_storage_type format, T* data)
    : format_(format), data_(data)
  { assert(format == array_format);}

  /// This constructor is provided so that `User_storage<T>` and
  /// `User_storage<complex<T> >` can be interchanged, however it
  /// should not be called.
  User_storage(user_storage_type format, T* real, T* /*imag*/)
    : format_(format), data_(real)
  { assert(0);}

  user_storage_type format() const { return format_;}

  T    get(index_type i) { return this->data_[i];}
  void put(index_type i, T val) { this->data_[i] = val;}

  /// Return the user storage in a format acceptable for initializing
  /// a Storage class.
  template <storage_format_type C>
  typename Storage<C, T>::type as() const
  {
    return User_storage_as_helper<T, C>::exec(*this);
  }

  typename Storage<interleaved_complex, T>::type 
  as_interleaved() const
  {
    assert(this->format_ == array_format);
    return this->data_;
  }

  // For scalar types, there is no distinction between interleaved and
  // split.
  typename Storage<split_complex, T>::type
  as_split() const
  {
    assert(this->format_ == array_format);
    return this->data_;
  }

  void find(T *&pointer) { pointer = this->data_;}
  void rebind(T *pointer) { this->data_ = pointer;}

private:
  user_storage_type format_;
  T* data_;
};



/// User_storage specialization for complex types.
///
/// Can store user-storage in array format, interleaved format, or
/// split format.
template <typename T>
class User_storage<complex<T> >
{
public:
  User_storage()
    : format_(no_user_format)
  {
    // Zero everything out so that find() returns NULL values.
    this->u_.data_ = 0;
    this->u_.split_.real_ = 0;
    this->u_.split_.imag_ = 0;
  }

  User_storage(user_storage_type format, complex<T> *data)
    : format_(format)
  {
    assert(format == array_format);
    this->u_.data_ = data;
  }

  User_storage(user_storage_type format, T *real, T *imag)
    : format_(format)
  { 
    assert(format == interleaved_format || format == split_format);
    this->u_.split_.real_ = real;
    this->u_.split_.imag_ = imag;
  }

  user_storage_type format() const { return this->format_;}

  complex<T> get(index_type i)
  {
    assert(this->format_ != no_user_format);

    if (this->format_ == array_format)
      return this->u_.data_[i];
    else if (this->format_ == interleaved_format)
      return complex<T>(this->u_.split_.real_[2*i+0],
			this->u_.split_.real_[2*i+1]);
    else // if (format_ == split_format)
      return complex<T>(this->u_.split_.real_[i],
			this->u_.split_.imag_[i]);
  }

  void put(index_type i, complex<T> val)
  {
    assert(this->format_ != no_user_format);

    if (this->format_ == array_format)
      this->u_.data_[i] = val;
    else if (this->format_ == interleaved_format)
    {
      this->u_.split_.real_[2*i+0] = val.real();
      this->u_.split_.real_[2*i+1] = val.imag();
    }
    else // if (format_ == split_format)
    {
      this->u_.split_.real_[i] = val.real();
      this->u_.split_.imag_[i] = val.imag();
    }
  }

  template <storage_format_type C>
  typename Storage<C, complex<T> >::type as() const
  {
    return User_storage_as_helper<complex<T>, C>::exec(*this);
  }

  /// Return the user storage in a format acceptable for initializing
  /// a Storage class with array/interleaved format, or return NULL.
  typename Storage<interleaved_complex, complex<T> >::type
  as_interleaved() const
  {
    assert(this->format_ != no_user_format);

    if (this->format_ == array_format)
      return this->u_.data_;
    else if (this->format_ == interleaved_format)
      return (complex<T>*)this->u_.split_.real_;
    else // if (format_ == split_format)
      return NULL;
  }

  /// Return the user storage in a format acceptable for initializing
  /// a Storage class with split format, or return effective NULL.
  typename Storage<split_complex, complex<T> >::type
  as_split() const
  {
    assert(this->format_ != no_user_format);

    if (this->format_ == array_format)
      return std::pair<T*, T*>(0, 0);
    else if (this->format_ == interleaved_format)
      return std::pair<T*, T*>(0, 0);
    else // if (format_ == split_format)
      return std::pair<T*, T*>(this->u_.split_.real_, this->u_.split_.imag_);
  }

  void find(complex<T> *&pointer) { pointer = this->u_.data_;}
  void find(T *&pointer) { pointer = this->u_.split_.real_;}
  void find(T *&real_pointer, T *&imag_pointer)
  {
    real_pointer = this->u_.split_.real_;
    imag_pointer = this->u_.split_.imag_;
  }

  void rebind(complex<T> *pointer)
  {
    this->u_.data_ = pointer;
    this->format_  = array_format;
  }

  void rebind(T *pointer)
  {
    this->u_.split_.real_ = pointer; 
    this->format_         = interleaved_format;
  }

  void rebind(T *real_pointer, T *imag_pointer)
  {
    this->u_.split_.real_ = real_pointer;
    this->u_.split_.imag_ = imag_pointer;
    this->format_         = imag_pointer ? split_format : interleaved_format;
  }

private:
  user_storage_type format_;
  union 
  {
    complex<T> *data_;
    struct 
    {
      T *real_;
      T *imag_;
    } split_;
  } u_;
};

} // namespace vsip::impl
} // namespace vsip

#endif

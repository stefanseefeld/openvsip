//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_storage_host_hpp_
#define ovxx_storage_host_hpp_

#include <ovxx/storage/storage.hpp>
#include <ovxx/allocator.hpp>

namespace ovxx
{

template <typename T, storage_format_type F>
class host_storage : public storage<T, F>
{
public:
  typedef typename storage<T, F>::ptr_type ptr_type;

  host_storage(length_type size, bool allocate=true)
    : storage<T,F>(size),
      allocator_(allocator::get_default()),
      own_data_(true)
  {
    if (allocate) this->allocate();
  }
  host_storage(length_type size, ptr_type buffer)
    : storage<T,F>(size),
      allocator_(allocator::get_default()),
      own_data_(!buffer)
  {
    if (!own_data_)
      this->data_ = buffer;
    else this->allocate();
  }
  host_storage(allocator *a, length_type size, bool allocate=true)
    : storage<T,F>(size),
      allocator_(a),
      own_data_(true)
  {
    if (allocate) this->allocate();
  }
  host_storage(allocator *a, length_type size, ptr_type buffer = ptr_type())
    : storage<T,F>(size),
      allocator_(a),
      own_data_(!buffer)
  {
    if (!own_data_)
      this->data_ = buffer;
    else this->allocate();
  }
  virtual ~host_storage()
  {
    if (own_data_)
      deallocate();
  }
  virtual void allocate()
  {
    if (own_data_ && !this->data_)
    {
      this->data_ = allocator_->allocate<T>(this->size_);
      OVXX_TRACE("host_storage::allocate(%d)", this->size_*sizeof(T));
    }
  }
  virtual void deallocate()
  {
    OVXX_TRACE("host_storage::deallocate(%d)", this->size_*sizeof(T));
    allocator_->deallocate(this->data_, this->size_);
    this->data_ = ptr_type();
  }
  virtual void resize(length_type size)
  {
    if (own_data_)
    {
      if (this->size_ == size) return;
      else deallocate();
    }
    else
      this->data_ = ptr_type();
    this->size_ = size;
    own_data_ = true;
    allocate();
  }
  void resize(length_type size, ptr_type buffer)
  {
    if (buffer)
    {
      if (own_data_) deallocate();
      this->data_ = buffer;
      this->size_ = size;
      own_data_ = false;
    }
    else resize(size);
  }

private:
  allocator *allocator_;
  bool own_data_;
};

template <typename T>
class host_storage<complex<T>, interleaved_complex>
  : public storage<complex<T>, interleaved_complex>
{
public:
  typedef typename storage<complex<T>, interleaved_complex>::ptr_type ptr_type;

  host_storage(length_type size, bool allocate=true)
    : storage<complex<T>, interleaved_complex>(size),
      allocator_(allocator::get_default()),
      own_data_(true)
  {
    if (allocate) this->allocate();
  }
  host_storage(length_type size, ptr_type buffer = ptr_type())
    : storage<complex<T>, interleaved_complex>(size),
      allocator_(allocator::get_default()),
      own_data_(!buffer)
  {
    if (!own_data_)
      this->data_ = buffer;
    else this->allocate();
  }
  host_storage(allocator *a, length_type size, bool allocate=true)
    : storage<complex<T>, interleaved_complex>(size),
      allocator_(a),
      own_data_(true)
  {
    if (allocate) this->allocate();
  }
  host_storage(allocator *a, length_type size, ptr_type buffer = ptr_type())
    : storage<complex<T>, interleaved_complex>(size),
      allocator_(a),
      own_data_(!buffer)
  {
    if (!own_data_)
      this->data_ = buffer;
    else this->allocate();
  }
  virtual ~host_storage()
  {
    if (own_data_)
      deallocate();
  }
  virtual void allocate()
  {
    if (own_data_ && !this->data_)
      this->data_ = allocator_->allocate<T>(2*this->size_);
  }
  virtual void deallocate()
  {
    allocator_->deallocate(this->data_, 2*this->size_);
    this->data_ = ptr_type();
  }
  virtual void resize(length_type size)
  {
    if (own_data_)
    {
      if (this->size_ == size) return;
      else deallocate();
    }
    else
      this->data_ = ptr_type();
    this->size_ = size;
    own_data_ = true;
    allocate();
  }
  void resize(length_type size, ptr_type buffer)
  {
    if (buffer)
    {
      if (own_data_) deallocate();
      this->data_ = buffer;
      this->size_ = size;
      own_data_ = false;
    }
    else resize(size);
  }

private:
  allocator *allocator_;
  bool own_data_;
};

template <typename T>
class host_storage<complex<T>, split_complex>
  : public storage<complex<T>, split_complex>
{
public:
  typedef typename storage<complex<T>, split_complex>::ptr_type ptr_type;

  host_storage(length_type size, bool allocate=true)
    : storage<complex<T>, split_complex>(size),
      allocator_(allocator::get_default()),
      own_data_(true)
  {
    if (allocate) this->allocate();
  }
  host_storage(length_type size, ptr_type buffer = ptr_type())
    : storage<complex<T>, split_complex>(size),
      allocator_(allocator::get_default()),
      own_data_(!buffer.first)
  {
    if (!own_data_)
      this->data_ = buffer;
    else this->allocate();
  }
  host_storage(allocator *a, length_type size, bool allocate=true)
    : storage<complex<T>, split_complex>(size),
      allocator_(a),
      own_data_(true)
  {
    if (allocate) this->allocate();
  }
  host_storage(allocator *a, length_type size, ptr_type buffer = ptr_type())
    : storage<complex<T>, split_complex>(size),
      allocator_(a),
      own_data_(!buffer.first)
  {
    if (!own_data_)
      this->data_ = buffer;
    else this->allocate();
  }
  virtual ~host_storage()
  {
    if (own_data_)
      deallocate();
  }
  virtual void allocate()
  {
    if (own_data_ && !this->data_.first)
    {
      this->data_.first = allocator_->allocate<T>(this->size_);
      this->data_.second = allocator_->allocate<T>(this->size_);
    }
  }
  virtual void deallocate()
  {
    allocator_->deallocate(this->data_.second, this->size_);
    allocator_->deallocate(this->data_.first, this->size_);
    this->data_ = ptr_type();
  }
  virtual void resize(length_type size)
  {
    if (own_data_)
    {
      if (this->size_ == size) return;
      else deallocate();
    }
    else
      this->data_ = ptr_type();
    this->size_ = size;
    own_data_ = true;
    allocate();
  }
  void resize(length_type size, ptr_type buffer)
  {
    if (buffer.first)
    {
      if (own_data_) deallocate();
      this->data_ = buffer;
      this->size_ = size;
      own_data_ = false;
    }
    else resize(size);
  }
private:
  allocator *allocator_;
  bool own_data_;
};

template <typename T>
class host_storage<complex<T>, any_storage_format>
  : public storage<complex<T>, any_storage_format>
{
public:
  typedef typename storage<complex<T>, any_storage_format>::ptr_type ptr_type;

  host_storage(length_type size, storage_format_type f)
    : storage<complex<T>, any_storage_format>(size),
      allocator_(allocator::get_default()),
      format_(f),
      own_data_(true)
  {
    this->allocate();
  }
  host_storage(length_type size, ptr_type buffer)
    : storage<complex<T>, any_storage_format>(size),
      allocator_(allocator::get_default()),
      format_(buffer.format()),
      own_data_(!buffer)
  {
    if (!own_data_)
      this->data_ = buffer;
    else this->allocate();
  }
  host_storage(allocator *a, length_type size, storage_format_type f)
    : storage<complex<T>, any_storage_format>(size),
      allocator_(a),
      format_(f),
      own_data_(true)
  {
    this->allocate();
  }
  host_storage(allocator *a, length_type size, ptr_type buffer)
    : storage<complex<T>, any_storage_format>(size),
      allocator_(a),
      format_(buffer.format()),
      own_data_(!buffer)
  {
    if (!own_data_)
      this->data_ = buffer;
    else
      this->allocate();
  }
  virtual ~host_storage()
  {
    if (own_data_)
      deallocate();
  }
  virtual void allocate()
  {
    switch (format_)
    {
      case array:
	this->data_ = allocator_->allocate<complex<T> >(this->size_);
	break;
      case interleaved_complex:
	this->data_ = allocator_->allocate<T>(2*this->size_);
	break;
      case split_complex:
	this->data_ = std::make_pair(allocator_->allocate<T>(this->size_),
				     allocator_->allocate<T>(this->size_));
	break;
      default: assert(0);
    }
  }
  virtual void deallocate()
  {
    switch (format_)
    {
      case array:
      {
	complex<T> *ptr = this->data_.template as<array>();
	allocator_->deallocate(ptr, this->size_);
	break;
      }
      case interleaved_complex:
      {
	T *ptr = this->data_.template as<interleaved_complex>();
	allocator_->deallocate(ptr, 2*this->size_);
	break;
      }
      case split_complex:
      {
	std::pair<T*,T*> ptr = this->data_.template as<split_complex>();
	allocator_->deallocate(ptr.second, this->size_);
	allocator_->deallocate(ptr.first, this->size_);
	break;
      }
      default: assert(0);
    }
    this->data_ = ptr_type();
  }
  virtual void resize(length_type size)
  {
    if (own_data_)
    {
      if (this->size_ == size) return;
      else deallocate();
    }
    else
      this->data_ = ptr_type();
    this->size_ = size;
    allocate();
    own_data_ = true;
  }
  void resize(length_type size, ptr_type buffer)
  {
    if (buffer)
    {
      if (own_data_) deallocate();
      this->data_ = buffer;
      this->size_ = size;
      own_data_ = false;
    }
    else
      resize(size);
  }

  storage_format_type format() const { return format_;}

private:
  allocator *allocator_;
  storage_format_type format_;
  bool own_data_;
};

} // namespace ovxx

#endif

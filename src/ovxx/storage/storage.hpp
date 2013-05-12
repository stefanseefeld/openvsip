//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_storage_storage_hpp_
#define ovxx_storage_storage_hpp_

#include <ovxx/support.hpp>
#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/storage/traits.hpp>

namespace ovxx
{

template <typename T, storage_format_type F>
class storage : detail::noncopyable
{
  typedef storage_traits<T, F> t;
public:
  typedef typename t::value_type value_type;
  typedef typename t::ptr_type ptr_type;
  typedef typename t::const_ptr_type const_ptr_type;
  typedef typename t::reference_type reference_type;
  typedef typename t::const_reference_type const_reference_type;

  storage(length_type size) : size_(size), data_(), valid_(true) {}
  virtual ~storage() {}
  virtual void allocate() = 0;
  virtual void deallocate() = 0;
  bool is_allocated() { return !t::is_null(data_);}
  virtual void resize(length_type size) = 0;

  void invalidate() { valid_ = false;}
  void validate() { valid_ = true;}
  bool is_valid() const { return valid_;}

  ptr_type ptr() { return data_;}
  const_ptr_type ptr() const { return data_;}
  length_type size() const { return size_;}

  value_type get(index_type i) const { return t::get(data_, i);}
  void put(index_type i, value_type v) { t::put(data_, i, v);}
  reference_type at(index_type i) { return t::at(data_, i);}
  const_reference_type at(index_type i) const { return t::at(data_, i);}

protected:
  length_type size_;
  ptr_type data_;
  bool valid_;
};

} // namespace ovxx

#endif

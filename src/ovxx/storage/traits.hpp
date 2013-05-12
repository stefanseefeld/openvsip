//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_storage_traits_hpp_
#define ovxx_storage_traits_hpp_

#include <ovxx/support.hpp>

namespace vsip
{
enum user_storage_type
{
  no_user_format = 0,
  array_format,
  split_format,
  interleaved_format
};
}

namespace ovxx
{

template <typename T, storage_format_type F> struct storage_traits;

template <typename T>
struct storage_traits<T, array>
{
  typedef T value_type;
  typedef T *ptr_type;
  typedef T const *const_ptr_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;

  static bool is_null(const_ptr_type p) { return !p;}
  static ptr_type offset(ptr_type ptr, index_type i)
  { return ptr + i;}
  static value_type get(const_ptr_type ptr, index_type i)
  { return ptr[i];}
  static void put(ptr_type ptr, index_type i, value_type v)
  { ptr[i] = v;}
  static reference_type at(ptr_type ptr, index_type i)
  { return ptr[i];}
  static const_reference_type at(const_ptr_type ptr, index_type i)
  { return ptr[i];}

  static bool is_compatible(user_storage_type s)
  { return s == array_format || s == interleaved_format;}
};

template <typename T>
struct storage_traits<complex<T>, interleaved_complex>
{
  typedef complex<T> value_type;
  typedef T *ptr_type;
  typedef T const *const_ptr_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;

  static bool is_null(const_ptr_type p) { return !p;}
  static ptr_type offset(ptr_type ptr, index_type i)
  { return ptr + 2*i;}
  static value_type get(const_ptr_type ptr, index_type i)
  { return value_type(ptr[2*i],ptr[2*i+1]);}
  static void put(ptr_type ptr, index_type i, value_type v)
  { ptr[2*i] = v.real(), ptr[2*i+1] = v.imag();}
  static reference_type at(ptr_type ptr, index_type i)
  { return reinterpret_cast<complex<T>*>(ptr)[i];}
  static const_reference_type at(const_ptr_type ptr, index_type i)
  { return reinterpret_cast<complex<T> const*>(ptr)[i];}
  static bool is_compatible(user_storage_type s)
  { return s == array_format || s == interleaved_format;}
};

template <typename T>
struct storage_traits<complex<T>, split_complex>
{
  typedef complex<T> value_type;
  typedef std::pair<T*,T*> ptr_type;
  typedef std::pair<T const *,T const *> const_ptr_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;

  static bool is_null(const_ptr_type p) { return !p.first;}
  static ptr_type offset(ptr_type ptr, index_type i)
  { return std::make_pair(ptr.first + i, ptr.second + i);}
  static value_type get(const_ptr_type ptr, index_type i)
  { return value_type(ptr.first[i],ptr.second[i]);}
  static void put(ptr_type ptr, index_type i, value_type v)
  { ptr.first[i] = v.real(), ptr.second[i] = v.imag();}
  static bool is_compatible(user_storage_type s)
  { return s == split_format;}
};

namespace detail
{
template <typename T, storage_format_type F>
struct storage_cast;

template <typename T>
struct storage_cast<T, array>
{
  template <typename O>
  static typename storage_traits<T, array>::const_ptr_type
  exec(O const &o) { return o.as_array();}
  template <typename O>
  static typename storage_traits<T, array>::ptr_type
  exec(O &o) { return o.as_array();}
};

template <typename T>
struct storage_cast<T, interleaved_complex>
{
  template <typename O>
  static typename storage_traits<T, interleaved_complex>::const_ptr_type
  exec(O const &o) { return o.as_interleaved();}
  template <typename O>
  static typename storage_traits<T, interleaved_complex>::ptr_type
  exec(O &o) { return o.as_interleaved();}
};

template <typename T>
struct storage_cast<T, split_complex>
{
  template <typename O>
  static typename storage_traits<T, split_complex>::const_ptr_type
  exec(O const &o) { return o.as_split();}
  template <typename O>
  static typename storage_traits<T, split_complex>::ptr_type
  exec(O &o) { return o.as_split();}
};

} // namespace ovxx::detail

template <typename T> class pointer;
template <typename T> class const_pointer;

template <typename T>
struct storage_traits<complex<T>, any_storage_format>
{
  typedef complex<T> value_type;
  typedef pointer<complex<T> > ptr_type;
  typedef const_pointer<complex<T> > const_ptr_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;

  static bool is_null(const_ptr_type p) { return !p;}
  static ptr_type offset(ptr_type ptr, index_type i)
  {
    switch (ptr.format())
    {
      case array: return ptr_type(ptr.template as<array>() + i);
      case interleaved_complex:
	return ptr_type(ptr.template as<interleaved_complex>() + 2*i);
      case split_complex:
      {
	std::pair<T*,T*> p = ptr.template as<split_complex>();
	return ptr_type(p.first + i, p.second + i);
      }
      default: assert(0);
    }
  }
  static const_ptr_type offset(const_ptr_type ptr, index_type i)
  {
    switch (ptr.format())
    {
      case array: return const_ptr_type(ptr.template as<array>() + i);
      case interleaved_complex:
	return const_ptr_type(ptr.template as<interleaved_complex>() + 2*i);
      case split_complex:
      {
	std::pair<T const*,T const*> p = ptr.template as<split_complex>();
	return const_ptr_type(p.first + i, p.second + i);
      }
      default: assert(0);
    }
  }
  static value_type get(const_ptr_type ptr, index_type i)
  {
    switch (ptr.format())
    {
      case array: return *ptr.template as<array>();
      case interleaved_complex:
	return *reinterpret_cast<value_type*>(ptr.template as<interleaved_complex>());
      case split_complex:
      {
	std::pair<T const*,T const*> p = ptr.template as<split_complex>();
	return value_type(p.first, p.second);
      }
      default: assert(0);
    }
  }
  static void put(ptr_type ptr, index_type i, value_type v)
  {
    switch (ptr.format())
    {
      case array:
	*ptr.template as<array>() = v;
	break;
      case interleaved_complex:
	*reinterpret_cast<value_type*>(ptr.template as<interleaved_complex>()) = v;
	break;
      case split_complex:
      {
	std::pair<T const*,T const*> p = ptr.template as<split_complex>();
	p.first = v.real();
	p.second = v.imag();
	break;
      }
      default: assert(0);
    }
  }
};

} // namespace ovxx

#endif

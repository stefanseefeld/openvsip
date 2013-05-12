//
// Copyright (c) 2010 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_pointer_hpp_
#define ovxx_pointer_hpp_

#include <ovxx/storage.hpp>
#include <ovxx/c++11.hpp>

namespace ovxx
{

/// Class to represent a pointer to data whose storage format
/// isn't known at compile-time.
template <typename T> class pointer;

template <storage_format_type F, typename T>
typename storage_traits<T, F>::ptr_type
storage_cast(pointer<T> &p)
{ return detail::storage_cast<T, F>::exec(p);}

template <storage_format_type F, typename T>
typename storage_traits<T, F>::const_ptr_type
storage_cast(pointer<T> const &p)
{ return detail::storage_cast<T, F>::exec(p);}

template <storage_format_type F, typename T>
typename storage_traits<T, F>::const_ptr_type
storage_cast(const_pointer<T> const &p)
{ return detail::storage_cast<T, F>::exec(p);}

template <typename T>
class pointer<complex<T> >
{
  template <typename, storage_format_type>
  friend struct detail::storage_cast;
public:
  pointer() : format_(any_storage_format) { data_.a = 0;}
  pointer(complex<T> *ptr) : format_(array) { data_.a = ptr;}
  pointer(T *ptr) : format_(interleaved_complex) { data_.i = ptr;}
  pointer(std::pair<T*, T*> const &ptr) : format_(split_complex)
  { data_.s.first = ptr.first, data_.s.second = ptr.second;}
  pointer(storage_format_type f) : format_(f)
  {
    if (format_ == array || format_ == any_storage_format) data_.a = 0;
    else if (format_ == interleaved_complex) data_.i = 0;
    else if (format_ == split_complex) data_.s.first = 0;
  }

  operator bool()
  {
    switch (format_)
    {
      case interleaved_complex: return data_.i != 0;
      case split_complex: return data_.s.first != 0;
      default: return data_.a != 0;
    }
  }
  storage_format_type format() const { return format_;}

  template <storage_format_type F>
  typename storage_traits<complex<T>, F>::ptr_type as()
  { return storage_cast<F>(*this);}
  template <storage_format_type F>
  typename storage_traits<complex<T>, F>::const_ptr_type as() const
  { return storage_cast<F>(*this);}

private:
  complex<T> *as_array() const 
  {
    OVXX_PRECONDITION(format_ == array || format_ == interleaved_complex);
    return data_.a;
  }
  T *as_interleaved() const
  {
    OVXX_PRECONDITION(format_ == array || format_ == interleaved_complex);
    return data_.i;
  }
  std::pair<T*, T*> as_split() const
  {
    OVXX_PRECONDITION(format_ == split_complex);
    return std::pair<T*,T*>(data_.s.first, data_.s.second);
  }

  storage_format_type format_;
  union 
  {
    complex<T> *a;
    T *i;
    // std::pair<T*,T*> is only allowed since C++11 here.
    struct { T*first, *second;} s;
  } data_;
};

template <typename T>
class const_pointer<complex<T> >
{
  template <typename, storage_format_type>
  friend struct detail::storage_cast;
public:
  const_pointer() : format_(any_storage_format) { data_.a = 0;}
  const_pointer(complex<T> const *ptr) : format_(array) { data_.a = ptr;}
  const_pointer(T const *ptr) : format_(interleaved_complex) { data_.i = ptr;}
  const_pointer(std::pair<T const *, T const *> const &ptr)
    : format_(split_complex) { data_.s.first = ptr.first, data_.s.second = ptr.second;}
  const_pointer(pointer<complex<T> > const &ptr)
    : format_(ptr.format()) 
  {
    switch (format_)
    {
      case array:
      case interleaved_complex:
	data_.a = ptr.template as<array>();
	break;
      case split_complex:
      {
	std::pair<T const*,T const*> p = ptr.template as<split_complex>();
	data_.s.first = p.first;
	data_.s.second = p.second;
	break;
      }
      default:
	data_.a = 0;
    }
  }
  const_pointer(storage_format_type f) : format_(f)
  {
    if (format_ == array || format_ == any_storage_format) data_.a = 0;
    else if (format_ == interleaved_complex) data_.i = 0;
    else if (format_ == split_complex) data_.s.first = 0;
  }

  operator bool()
  {
    if (format_ == array || format_ == any_storage_format)
      return data_.a != 0;
    else if (format_ == interleaved_complex)
      return data_.i != 0;
    else if (format_ == split_complex)
      return data_.s.first != 0;
  }

  storage_format_type format() const { return format_;}

  template <storage_format_type F>
  typename storage_traits<complex<T>, F>::const_ptr_type as()
  { return storage_cast<F>(*this);}

private:
  complex<T> const *as_array() const { return data_.a;}
  T const *as_interleaved() const { return data_.i;}
  std::pair<T const*, T const*> as_split() const 
  { return std::pair<T const*,T const*>(data_.s.first, data_.s.second);}

  storage_format_type format_;
  union 
  {
    complex<T> const *a;
    T const *i;
    // std::pair<T*,T*> is only allowed since C++11 here.
    struct { T const*first, *second;} s;
  } data_;
};

namespace detail
{

template <typename T> 
struct Const_cast
{
  static T *cast(T *p) { return p;}
};

template <typename T>
struct Const_cast<T*> 
{
  static T *cast(T *p) { return p;}
  static T *cast(T const*p) { return const_cast<T*>(p);}
};

template <typename T>
struct Const_cast<std::pair<T*,T*> >
{
  static std::pair<T*,T*> cast(std::pair<T*,T*> p) { return p;}
  static std::pair<T*,T*> cast(std::pair<T const*,T const*> p)
  { return std::make_pair(const_cast<T*>(p.first), const_cast<T*>(p.second));}
};

template <typename T>
struct Const_cast<pointer<complex<T> > >
{
  static pointer<complex<T> > cast(pointer<complex<T> > p) { return p;}
  static pointer<complex<T> > cast(const_pointer<complex<T> > p)
  {
    typedef complex<T> C;
    switch (p.format())
    {
      case array:
	return pointer<C>(const_cast<C*>(p.template as<array>()));
      case interleaved_complex:
	return pointer<C>(const_cast<T*>(p.template as<interleaved_complex>()));
      case split_complex:
      {
	std::pair<T const*,T const*> pp(p.template as<split_complex>());
	return pointer<C>(std::make_pair(const_cast<T*>(pp.first),
					 const_cast<T*>(pp.second)));
      }
      default:
	return pointer<C>();
    }
  }
};

template <typename T>
struct pointer_cast
{
  static T cast(T ptr) { return ptr;}
};

template <typename T>
struct pointer_cast<complex<T> *>
{
  static complex<T> *cast(complex<T> *ptr) { return ptr;}
  static complex<T> *cast(T *ptr) { return reinterpret_cast<complex<T> *>(ptr);}
  static complex<T> *cast(pointer<complex<T> > &ptr) { return ptr.template as<array>();}
};

template <typename T>
struct pointer_cast<complex<T> const *>
{
  static complex<T> const *cast(complex<T> const *ptr) { return ptr;}
  static complex<T> const *cast(T const *ptr) { return reinterpret_cast<complex<T> const*>(ptr);}
  static complex<T> const *cast(const_pointer<complex<T> > const &ptr) { return ptr.template as<array>();}
};

template <typename T>
struct pointer_cast<T *>
{
  static T *cast(T *ptr) { return ptr;}
  static T *cast(complex<T> *ptr) { return reinterpret_cast<T *>(ptr);}
  static T *cast(pointer<complex<T> > &ptr) { return ptr.template as<interleaved_complex>();}
};

template <typename T>
struct pointer_cast<T const *>
{
  static T const *cast(T const *ptr) { return ptr;}
  static T const *cast(complex<T> const *ptr) { return reinterpret_cast<T const*>(ptr);}
  static T const *cast(const_pointer<complex<T> > const &ptr) { return ptr.template as<interleaved_complex>();}
};

} // namespace ovxx::detail

template <typename T1, typename T2>
inline T1
const_cast_(T2 p) { return detail::Const_cast<T1>::cast(p);}

template <typename T1, typename T2>
inline T1
pointer_cast(T2 p) { return detail::pointer_cast<T1>::cast(p);}

template <typename T>
inline bool
is_aligned_to(T* pointer, std::size_t align)
{
  return reinterpret_cast<std::size_t>(pointer) % align == 0;
}

template <typename T>
inline bool
is_aligned_to(std::pair<T*, T*> pointer, std::size_t align)
{
  return reinterpret_cast<std::size_t>(pointer.first)  % align == 0 &&
    reinterpret_cast<std::size_t>(pointer.second) % align == 0;
}

} // namespace ovxx

#endif

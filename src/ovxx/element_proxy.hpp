//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_element_proxy_hpp_
#define ovxx_element_proxy_hpp_

#include <ovxx/support.hpp>
#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/block_traits.hpp>
#include <ostream>

namespace ovxx
{
template <typename B, dimension_type D>
class element_proxy
{
public:
  typedef B block_type;
  typedef typename B::value_type value_type;

  element_proxy(block_type &b, Index<D> const &i)
    : block_(b), index_(i) {};

  // support value = view(i)
  operator value_type() const 
  { return ovxx::get(block_, index_);}

  // support view(i) = value
  element_proxy &operator=(value_type v)
  {
    ovxx::put(block_, index_, v);
    return *this;
  }

  // support view1(i) = view2(j)
  element_proxy &operator=(element_proxy &other)
  { return *this = value_type(other);}

  template <typename T>
  bool operator==(T v) const { return static_cast<value_type>(*this) == v;}
  bool operator==(element_proxy const &other) const 
  { return static_cast<value_type>(*this) == static_cast<value_type>(other);}

  value_type operator+=(value_type v) 
  { return operator=(static_cast<value_type>(*this) + v);}
  value_type operator-=(value_type v)
  { return operator=(static_cast<value_type>(*this) - v);}
  value_type operator*=(value_type v)
  { return operator=(static_cast<value_type>(*this) * v);}
  value_type operator/=(value_type v)
  { return operator=(static_cast<value_type>(*this) / v);}

private:
  block_type &block_;
  Index<D> index_;
};

template <typename B, dimension_type D>
typename B::value_type &operator+= (typename B::value_type &t, element_proxy<B, D> const &p) 
{ return t += static_cast<typename B::value_type>(p);}
template <typename B, dimension_type D>
typename B::value_type operator+ (typename B::value_type t, element_proxy<B, D> const &p) 
{ return t + static_cast<typename B::value_type>(p);}
template <typename B, dimension_type D>
typename B::value_type operator+ (element_proxy<B, D> const &p, typename B::value_type t) 
{ return static_cast<typename B::value_type>(p) + t;}
template <typename B, dimension_type D>
typename B::value_type operator+ (element_proxy<B, D> const &p1, element_proxy<B, D> const &p2) 
{ return static_cast<typename B::value_type>(p1) + static_cast<typename B::value_type>(p2);}

template <typename B, dimension_type D>
typename B::value_type &operator-= (typename B::value_type &t, element_proxy<B, D> const &p) 
{ return t -= static_cast<typename B::value_type>(p);}
template <typename B, dimension_type D>
typename B::value_type operator- (typename B::value_type t, element_proxy<B, D> const &p) 
{ return t - static_cast<typename B::value_type>(p);}
template <typename B, dimension_type D>
typename B::value_type operator- (element_proxy<B, D> const &p, typename B::value_type t) 
{ return static_cast<typename B::value_type>(p) - t;}
template <typename B, dimension_type D>
typename B::value_type operator- (element_proxy<B, D> const &p1, element_proxy<B, D> const &p2) 
{ return static_cast<typename B::value_type>(p1) - static_cast<typename B::value_type>(p2);}

template <typename B, dimension_type D>
typename B::value_type &operator*= (typename B::value_type &t, element_proxy<B, D> const &p) 
{ return t *= static_cast<typename B::value_type>(p);}
template <typename B, dimension_type D>
typename B::value_type operator* (typename B::value_type t, element_proxy<B, D> const &p) 
{ return t * static_cast<typename B::value_type>(p);}
template <typename B, dimension_type D>
typename B::value_type operator* (element_proxy<B, D> const &p, typename B::value_type t) 
{ return static_cast<typename B::value_type>(p) * t;}
template <typename B, dimension_type D>
typename B::value_type operator* (element_proxy<B, D> const &p1, element_proxy<B, D> const &p2) 
{ return static_cast<typename B::value_type>(p1) * static_cast<typename B::value_type>(p2);}

template <typename B, dimension_type D>
typename B::value_type &operator/= (typename B::value_type &t, element_proxy<B, D> const &p) 
{ return t /= static_cast<typename B::value_type>(p);}
template <typename B, dimension_type D>
typename B::value_type operator/ (typename B::value_type t, element_proxy<B, D> const &p) 
{ return t / static_cast<typename B::value_type>(p);}
template <typename B, dimension_type D>
typename B::value_type operator/ (element_proxy<B, D> const &p, typename B::value_type t) 
{ return static_cast<typename B::value_type>(p) / t;}
template <typename B, dimension_type D>
typename B::value_type operator/ (element_proxy<B, D> const &p1, element_proxy<B, D> const &p2) 
{ return static_cast<typename B::value_type>(p1) / static_cast<typename B::value_type>(p2);}
template <typename B, dimension_type D>
std::ostream &operator<< (std::ostream &os, element_proxy<B, D> const &p)
{ return os << static_cast<typename B::value_type>(p);}


template <typename B, dimension_type D>
struct proxy_factory : detail::noncopyable
{
  typedef element_proxy<B, D> reference_type;
  typedef element_proxy<B, D> const const_reference_type;

  static reference_type ref(B &b, index_type i)
  { return element_proxy<B, D>(b, Index<1>(i));}
  static reference_type ref(B &b, index_type i, index_type j)
  { return element_proxy<B, D>(b, Index<2>(i, j));}
  static reference_type ref(B &b, index_type i, index_type j, index_type k)
  { return element_proxy<B, D>(b, Index<3>(i, j, k));}
};

template <typename B>
struct ref_factory : detail::noncopyable
{
  typedef typename B::reference_type reference_type;
  typedef typename B::const_reference_type const_reference_type;

  static reference_type ref(B &b, index_type i)
  { return b.ref(i);}
  static reference_type ref(B &b, index_type i, index_type j)
  { return b.ref(i, j);}
  static reference_type ref(B &b, index_type i, index_type j, index_type k)
  { return b.ref(i, j, k);}
};

template <typename T>
struct is_element_proxy
{
  static bool const value = false;
  typedef T value_type;
};

template <typename B, dimension_type D>
struct is_element_proxy<element_proxy<B, D> >
{
  static bool const value = true;
  typedef typename B::value_type value_type;
};

} // namespace ovxx

#endif

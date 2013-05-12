//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef storage_util_hpp_
#define storage_util_hpp_

#include <iostream>
#include <cassert>

#include <ovxx/support.hpp>
#include <ovxx/storage/block.hpp>
#include <ovxx/domain_utils.hpp>
#include <ovxx/length.hpp>

using namespace ovxx;

template <typename O, dimension_type D>
index_type
to_index(Length<D> const &ext, Index<D> const &idx)
{
  if (D == 1)
    return idx[0];
  else if (D == 2)
    return idx[O::impl_dim0]*ext[O::impl_dim1]+idx[O::impl_dim1];
  else // if (D == 3)
    return idx[O::impl_dim0]*ext[O::impl_dim1]*ext[O::impl_dim2] +
           idx[O::impl_dim1]*ext[O::impl_dim2] + 
           idx[O::impl_dim2];
}
	  

template <typename O, typename T, dimension_type D, typename F>
void
fill_array_storage(T *data, Domain<D> const &dom, F func)
{
  Length<D> ext = extent(dom);
  for (Index<D> idx; valid(ext, idx); next(ext, idx))
  {
    index_type i = to_index<O>(ext, idx);
    data[i] = func(i);
  }
}

template <typename O, typename T, dimension_type D, typename F>
bool
check_array_storage(T *data, Domain<D> const &dom, F func)
{
  Length<D> ext = extent(dom);
  for (Index<D> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<O>(ext, idx);
    if (!equal(data[i], func(i)))
      return false;
  }
  return true;
}


template <typename O, typename T, dimension_type D, typename F>
void
fill_interleaved_storage(T *data, Domain<D> const &dom, F func)
{
  Length<D> ext = extent(dom);
  for (Index<D> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<O>(ext, idx);
    complex<T> val = func(i);
    data[2*i+0] = val.real();
    data[2*i+1] = val.imag();
  }
}

template <typename O, typename T, dimension_type D, typename F>
bool
check_interleaved_storage(T *data, Domain<D> const &dom, F func)
{
  Length<D> ext = extent(dom);
  for (Index<D> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<O>(ext, idx);
    complex<T> val = func(i);
    if (!equal(data[2*i+0], val.real()) ||
	!equal(data[2*i+1], val.imag()))
      return false;
  }
  return true;
}


template <typename O, typename T, dimension_type D, typename F>
void
fill_split_storage(std::pair<T*,T*> data, Domain<D> const &dom, F func)
{
  Length<D> ext = extent(dom);
  for (Index<D> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<O>(ext, idx);
    complex<T> val = func(i);
    data.first[i] = val.real();
    data.second[i] = val.imag();
  }
}


template <typename O, typename T, dimension_type D, typename F>
bool
check_split_storage(std::pair<T*,T*> data, Domain<D> const &dom, F func)
{
  Length<D> ext = extent(dom);
  for (Index<D> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<O>(ext, idx);
    complex<T> val = func(i);
    if (!equal(data.first[i], val.real()) ||
	!equal(data.second[i], val.imag()))
      return false;
  }
  return true;
}

template <typename O, typename B, dimension_type D, typename F>
void
fill_block(B &block, Domain<D> const &dom, F func)
{
  Length<D> ext = extent(dom);
  for (Index<D> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<O>(ext, idx);
    put(block, idx, func(i));
  }
}

template <typename O, typename B, dimension_type D, typename F>
bool
check_block(B &block, Domain<D> const &dom, F func)
{
  Length<D> ext = extent(dom);
  for (Index<D> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<O>(ext, idx);
    if (!equal(get(block, idx), func(i)))
      return false;
  }
  return true;
}

template <typename T>
class Filler
{
public:
  Filler(int k, int o) : k_(k), o_(o) {}
  T operator()(int i) { return T(k_*i+o_); }
private:
  int k_;
  int o_;
};

template <typename T>
class CFiller
{
public:
  CFiller(int k, int o1, int o2) : k_(k), o1_(o1), o2_(o2) {}
  complex<T> operator()(int i)
    { return complex<T>(T(k_*i+o1_), T(k_*i+o2_)); }
private:
  int k_;
  int o1_;
  int o2_;
};


#endif

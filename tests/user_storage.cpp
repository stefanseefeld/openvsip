//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/dense.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/length.hpp>
#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;

using vsip::impl::Length;



/***********************************************************************
  Definitions
***********************************************************************/
template <typename       Order,
	  dimension_type Dim>
index_type
to_index(Length<Dim> const& ext,
	 Index<Dim>  const& idx)
{
  if (Dim == 1)
    return idx[0];
  else if (Dim == 2)
    return idx[Order::impl_dim0]*ext[Order::impl_dim1]+idx[Order::impl_dim1];
  else // if (Dim == 3)
    return idx[Order::impl_dim0]*ext[Order::impl_dim1]*ext[Order::impl_dim2] +
           idx[Order::impl_dim1]*ext[Order::impl_dim2] + 
           idx[Order::impl_dim2];
}
	  

template <typename       Order,
	  typename       T,
	  dimension_type Dim,
	  typename       Func>
void
fill_array(
  T*                 data,
  Domain<Dim> const& dom,
  Func               fun)
{
  Length<Dim> ext = impl::extent(dom);
  for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
  {
    index_type i = to_index<Order>(ext, idx);
    data[i] = fun(i);
  }
}


template <typename       Order,
	  typename       T,
	  dimension_type Dim,
	  typename       Func>
bool
check_array(
  T*                 data,
  Domain<Dim> const& dom,
  Func               fun)
{
  Length<Dim> ext = impl::extent(dom);
  for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<Order>(ext, idx);
    if (!equal(data[i], fun(i)))
      return false;
  }
  return true;
}


template <typename       Order,
	  typename       T,
	  dimension_type Dim,
	  typename       Func>
void
fill_interleaved_array(
  T*                 data,
  Domain<Dim> const& dom,
  Func               fun)
{
  Length<Dim> ext = impl::extent(dom);
  for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<Order>(ext, idx);
    complex<T> val = fun(i);
    data[2*i+0] = val.real();
    data[2*i+1] = val.imag();
  }
}


template <typename       Order,
	  typename       T,
	  dimension_type Dim,
	  typename       Func>
bool
check_interleaved_array(
  T*                 data,
  Domain<Dim> const& dom,
  Func               fun)
{
  Length<Dim> ext = impl::extent(dom);
  for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<Order>(ext, idx);
    complex<T> val = fun(i);
    if (!equal(data[2*i+0], val.real()) ||
	!equal(data[2*i+1], val.imag()))
      return false;
  }
  return true;
}


template <typename       Order,
	  typename       T,
	  dimension_type Dim,
	  typename       Func>
void
fill_split_array(
  T*                 real,
  T*                 imag,
  Domain<Dim> const& dom,
  Func               fun)
{
  Length<Dim> ext = impl::extent(dom);
  for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<Order>(ext, idx);
    complex<T> val = fun(i);
    real[i] = val.real();
    imag[i] = val.imag();
  }
}


template <typename       Order,
	  typename       T,
	  dimension_type Dim,
	  typename       Func>
bool
check_split_array(
  T*                 real,
  T*                 imag,
  Domain<Dim> const& dom,
  Func               fun)
{
  Length<Dim> ext = impl::extent(dom);
  for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<Order>(ext, idx);
    complex<T> val = fun(i);
    if (!equal(real[i], val.real()) ||
	!equal(imag[i], val.imag()))
      return false;
  }
  return true;
}

template <typename       Order,
	  typename       Block,
	  dimension_type Dim,
	  typename       Func>
void
fill_block(
  Block&             block,
  Domain<Dim> const& dom,
  Func               fun)
{
  Length<Dim> ext = impl::extent(dom);
  for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<Order>(ext, idx);
    put(block, idx, fun(i));
  }
}

template <typename       Order,
	  typename       Block,
	  dimension_type Dim,
	  typename       Func>
bool
check_block(
  Block&             block,
  Domain<Dim> const& dom,
  Func               fun)
{
  Length<Dim> ext = impl::extent(dom);
  for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
  {
    index_type i = to_index<Order>(ext, idx);
    if (!equal(get(block, idx), fun(i)))
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



template <typename       T,
	  typename       Order,
	  dimension_type Dim,
	  typename       BlockT>
void
rebind_array(
  Domain<Dim> const& dom,
  BlockT&            block,
  int                k)
{
  length_type const size = block.size();

  T* data = new T[size];

  T* ptr;


  fill_array<Order>(data, dom, Filler<T>(k, 1));

  block.rebind(data);

  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == array_format); // rebind could change format

  block.find(ptr);
  test_assert(ptr == data);

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<Order>(block, dom, Filler<T>(k, 1)));

  fill_block<Order>(block, dom, Filler<T>(k+1, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_array<Order>(data, dom, Filler<T>(k+1, 1)));

  delete[] data;
}


template <typename       T,
	  typename       Order,
	  dimension_type Dim,
	  typename       BlockT>
void
rebind_split(
  Domain<Dim> const& dom,
  BlockT&            block,
  int                k)
{
  length_type const size = block.size();

  T* real = new T[size];
  T* imag = new T[size];

  T* real_ptr;
  T* imag_ptr;

  fill_split_array<Order>(real, imag, dom, CFiller<T>(k, 0, 1));

  block.rebind(real, imag);

  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == split_format); // rebind could change format

  block.find(real_ptr, imag_ptr);
  test_assert(real_ptr == real);
  test_assert(imag_ptr == imag);

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<Order>(block, dom, CFiller<T>(k, 0, 1)));
  fill_block<Order>(block, dom, CFiller<T>(k+1, 0, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_split_array<Order>(real, imag, dom, CFiller<T>(k+1, 0, 1)));

  delete[] real;
  delete[] imag;
}


template <typename       T,
	  typename       Order,
	  dimension_type Dim,
	  typename       BlockT>
void
rebind_interleaved(
  Domain<Dim> const& dom,
  BlockT&            block,
  int                k)
{
  length_type const size = block.size();

  T* data = new T[2*size];
  T* ptr;

  fill_interleaved_array<Order>(data, dom, CFiller<T>(k, 0, 1));

  block.rebind(data);
  
  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == interleaved_format);

  block.find(ptr);
  test_assert(ptr == data);

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<Order>(block, dom, CFiller<T>(k, 0, 1)));
  fill_block<Order>(block, dom, CFiller<T>(k+1, 0, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_interleaved_array<Order>(data, dom, CFiller<T>(k+1, 0, 1)));

  delete[] data;
}



// Test admit/release of array-format data.
//
// Requires (Template):
//   T is a block value type,
//   ORDER is a dimension-order tuple,
//   DIM is the block dimensionality (inferred from DOM)
//
// Requires (Arguments):
//   DOM is a domain indicating the block dimensions.

template <typename       T,
	  typename       Order,
	  dimension_type Dim>
void
test_array_format(
  Domain<Dim> const& dom)
{
  length_type const size = impl::size(dom);
  T* data = new T[size];
  T* ptr;

  Dense<Dim, T, Order> block(dom, data);

  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == array_format);

  // Check find()
  block.find(ptr);
  test_assert(ptr == data);

  fill_array<Order>(data, dom, Filler<T>(3, 0));

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<Order>(block, dom, Filler<T>(3, 0)));

  fill_block<Order>(block, dom, Filler<T>(3, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_array<Order>(data, dom, Filler<T>(3, 1)));

  // Check release with pointer
  block.admit(true);
  block.release(true, ptr);

  test_assert(ptr == data);

  delete[] data;

  for (int i=0; i<5; ++i)
  {
    rebind_array<T, Order>(dom, block, 5);
    rebind_array<T, Order>(dom, block, 6);
    rebind_array<T, Order>(dom, block, 7);
  }
}



template <typename       T,
	  typename       Order,
	  dimension_type Dim>
void
test_interleaved_format(
  Domain<Dim> const& dom)
{
  length_type const size = impl::size(dom);

  T* data = new T[2*size];
  T* ptr;

  Dense<Dim, complex<T>, Order> block(dom, data);

  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == interleaved_format);

  block.find(ptr);
  test_assert(ptr == data);

  fill_interleaved_array<Order>(data, dom, CFiller<T>(3, 0, 2));

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<Order>(block, dom, CFiller<T>(3, 0, 2)));
  fill_block<Order>(block, dom, CFiller<T>(3, 2, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_interleaved_array<Order>(data, dom, CFiller<T>(3, 2, 1)));

  // Check release with pointer
  block.admit(true);
  block.release(true, ptr);

  test_assert(ptr == data);

  delete[] data;

  for (int i=0; i<5; ++i)
  {
    rebind_interleaved<T, Order>(dom, block, 4);
    rebind_split<T, Order>      (dom, block, 5);
    rebind_split<T, Order>      (dom, block, 6);
    rebind_interleaved<T, Order>(dom, block, 7);
    rebind_interleaved<T, Order>(dom, block, 8);
  }
}


template <typename       T,
	  typename       Order,
	  dimension_type Dim>
void
test_split_format(
  Domain<Dim> const& dom)
{
  length_type const size = impl::size(dom);

  T* real = new T[size];
  T* imag = new T[size];

  T* real_ptr;
  T* imag_ptr;

  Dense<Dim, complex<T>, Order> block(dom, real, imag);

  test_assert(block.admitted()     == false);
  test_assert(block.user_storage() == split_format);

  block.find(real_ptr, imag_ptr);
  test_assert(real_ptr == real);
  test_assert(imag_ptr == imag);

  fill_split_array<Order>(real, imag, dom, CFiller<T>(3, 0, 2));

  block.admit(true);
  test_assert(block.admitted() == true);

  test_assert(check_block<Order>(block, dom, CFiller<T>(3, 0, 2)));
  fill_block<Order>(block, dom, CFiller<T>(3, 2, 1));

  block.release(true);
  test_assert(block.admitted() == false);

  test_assert(check_split_array<Order>(real, imag, dom, CFiller<T>(3, 2, 1)));

  // Check release with pointer
  block.admit(true);
  block.release(true, real_ptr, imag_ptr);

  test_assert(real_ptr == real);
  test_assert(imag_ptr == imag);

  delete[] real;
  delete[] imag;

  for (int i=0; i<5; ++i)
  {
    rebind_split<T, Order>      (dom, block, 4);
    rebind_interleaved<T, Order>(dom, block, 5);
    rebind_interleaved<T, Order>(dom, block, 6);
    rebind_split<T, Order>      (dom, block, 7);
    rebind_split<T, Order>      (dom, block, 8);
  }
}



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_array_format<float,          row1_type>(Domain<1>(50));
  test_array_format<complex<float>, row1_type>(Domain<1>(50));

  test_array_format<float,          row2_type>(Domain<2>(25, 50));
  test_array_format<complex<float>, row2_type>(Domain<2>(50, 25));
  test_array_format<float,          col2_type>(Domain<2>(25, 50));
  test_array_format<complex<float>, col2_type>(Domain<2>(50, 25));

  test_array_format<float, tuple<0, 1, 2> >(Domain<3>(25, 50, 13));
  test_array_format<float, tuple<0, 2, 1> >(Domain<3>(25, 50, 13));
  test_array_format<float, tuple<1, 0, 2> >(Domain<3>(25, 50, 13));
  test_array_format<float, tuple<2, 0, 1> >(Domain<3>(25, 50, 13));
  test_array_format<float, tuple<1, 2, 0> >(Domain<3>(25, 50, 13));
  test_array_format<float, tuple<2, 1, 0> >(Domain<3>(25, 50, 13));

  test_split_format<float, row1_type>(Domain<1>(50));
  test_split_format<float, row2_type>(Domain<2>(25, 35));
  test_split_format<float, col2_type>(Domain<2>(45, 15));
  test_split_format<float, tuple<0, 1, 2> >(Domain<3>(25, 50, 13));
  test_split_format<float, tuple<0, 2, 1> >(Domain<3>(25, 50, 13));
  test_split_format<float, tuple<1, 0, 2> >(Domain<3>(25, 50, 13));
  test_split_format<float, tuple<2, 0, 1> >(Domain<3>(25, 50, 13));
  test_split_format<float, tuple<1, 2, 0> >(Domain<3>(25, 50, 13));
  test_split_format<float, tuple<2, 1, 0> >(Domain<3>(25, 50, 13));

  test_interleaved_format<float, row1_type>(Domain<1>(50));
  test_interleaved_format<float, row2_type>(Domain<2>(25, 37));
  test_interleaved_format<float, col2_type>(Domain<2>(91, 13));
  test_interleaved_format<float, tuple<0, 1, 2> >(Domain<3>(25, 50, 13));
  test_interleaved_format<float, tuple<0, 2, 1> >(Domain<3>(25, 50, 13));
  test_interleaved_format<float, tuple<1, 0, 2> >(Domain<3>(25, 50, 13));
  test_interleaved_format<float, tuple<2, 0, 1> >(Domain<3>(25, 50, 13));
  test_interleaved_format<float, tuple<1, 2, 0> >(Domain<3>(25, 50, 13));
  test_interleaved_format<float, tuple<2, 1, 0> >(Domain<3>(25, 50, 13));
}

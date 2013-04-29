//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <cassert>

#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Vector coverage
***********************************************************************/

/// Sum of vector values.

template <typename T,
	  typename Block>
T
sum_view(const_Vector<T, Block> view)
{
  typedef T value_type;
  typedef typename vsip::impl::scalar_of<value_type>::type scalar_type;

  T sum = T();

  for (index_type i=0; i<view.size(); ++i)
    sum += view(i);

  return sum;
}



template <typename T,
	  typename Block>
void
check_view(Vector<T, Block> view)
{
  typedef T value_type;
  typedef typename vsip::impl::scalar_of<value_type>::type scalar_type;

  for (index_type i=0; i<view.size(); ++i)
    view(i) = value_type(1);

  T sum = sum_view(view);

  test_assert(equal(sum, value_type(view.size())));
}



template <typename T>
void
do_vector()
{
  Vector<T> vec(25);

  check_view(vec);
}



/***********************************************************************
  Matrix coverage
***********************************************************************/

template <typename T,
	  typename Block>
T
sum_view(const_Matrix<T, Block> view)
{
  typedef T value_type;
  typedef typename vsip::impl::scalar_of<value_type>::type scalar_type;

  T sum = T();

  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
      sum += view(i, j);

  return sum;
}



template <typename T,
	  typename Block>
void
check_view(Matrix<T, Block> view)
{
  typedef T value_type;
  typedef typename vsip::impl::scalar_of<value_type>::type scalar_type;

  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
      view(i, j) = value_type(1);

  T sum = sum_view(view);

  test_assert(equal(sum, value_type(view.size())));
}



template <typename T>
void
do_matrix()
{
  Matrix<T> mat(25, 32);

  check_view(mat);
}



/***********************************************************************
  Tensor coverage
***********************************************************************/

template <typename T,
	  typename Block>
T
sum_view(const_Tensor<T, Block> view)
{
  typedef T value_type;
  typedef typename vsip::impl::scalar_of<value_type>::type scalar_type;

  T sum = T();

  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
      for (index_type k=0; k<view.size(2); ++k)
	sum += view(i, j, k);

  return sum;
}



template <typename T,
	  typename Block>
void
check_view(Tensor<T, Block> view)
{
  typedef T value_type;
  typedef typename vsip::impl::scalar_of<value_type>::type scalar_type;

  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
      for (index_type k=0; k<view.size(2); ++k)
	view(i, j, k) = value_type(1);

  T sum = sum_view(view);

  test_assert(equal(sum, value_type(view.size())));
}



template <typename T>
void
do_tensor()
{
  Tensor<T> ten(25, 32, 8);

  check_view(ten);
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  do_vector<float>();
  do_vector<complex<float> >();

  do_matrix<float>();
  do_matrix<complex<float> >();

  do_tensor<float>();
  do_tensor<complex<float> >();
}

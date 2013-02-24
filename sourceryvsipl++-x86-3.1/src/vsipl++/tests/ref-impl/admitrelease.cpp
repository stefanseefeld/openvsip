/***********************************************************************

  File:   admitrelease.cpp
  Author: Jules Bergmann, CodeSourcery, LLC.
  Date:   02/17/2005

  Contents: Tests of the Admit/Release.

Copyright 2005 Georgia Tech Research Corporation, all rights reserved.

A non-exclusive, non-royalty bearing license is hereby granted to all
Persons to copy, distribute and produce derivative works for any
purpose, provided that this copyright notice and following disclaimer
appear on All copies: THIS LICENSE INCLUDES NO WARRANTIES, EXPRESSED
OR IMPLIED, WHETHER ORAL OR WRITTEN, WITH RESPECT TO THE SOFTWARE OR
OTHER MATERIAL INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE, OR ARISING
FROM A COURSE OF PERFORMANCE OR DEALING, OR FROM USAGE OR TRADE, OR OF
NON-INFRINGEMENT OF ANY PATENTS OF THIRD PARTIES. THE INFORMATION IN
THIS DOCUMENT SHOULD NOT BE CONSTRUED AS A COMMITMENT OF DEVELOPMENT
BY ANY OF THE ABOVE PARTIES.

The US Government has a license under these copyrights, and this
Material may be reproduced by or for the US Government.
  VSIPL++ Library

***********************************************************************/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cstdlib>
#include <vsip/initfin.hpp>
#include <vsip/dense.hpp>
#include <vsip/domain.hpp>
#include <vsip/support.hpp>

#include "test.hpp"



/***********************************************************************
  Class Declarations
***********************************************************************/

template <typename T1,
	  typename T2>
struct is_same
{ static bool const value = false; };

template <typename T>
struct is_same<T, T>
{ static bool const value = true; };



/***********************************************************************
  Function Definitions
***********************************************************************/

using vsip::Dense;
using vsip::Domain;
using vsip::length_type;
using vsip::index_type;
using vsip::array_format;
using vsip::split_format;
using vsip::interleaved_format;
using vsip::scalar_f;
using vsip::scalar_i;
using vsip::complex;

template <typename T>
void
test_1_scalar()
{
  length_type const size = 25;
  Domain<1>         dom(size);
  T*	data = new T[size];

  for (index_type i=0; i<size; ++i)
    data[i] = T(2*i+1);

  {
    Dense<1, T> block(dom, data);

    insist(block.size()         == size);
    insist(block.admitted()     == false);
    insist(block.user_storage() == array_format);

    block.admit(true);

    insist(block.admitted()     == true);
      
    for (index_type i=0; i<size; ++i)
      insist(equal(block.get(i), T(2*i+1)));

    for (index_type i=0; i<size; ++i)
      block.put(i, T(3*i+2));

    block.release(true);

    insist(block.admitted()     == false);
  }

  for (index_type i=0; i<size; ++i)
    insist(equal(data[i], T(3*i+2)));

  delete[] data;
}



template <typename T>
void
test_1_split()
{
  length_type const size = 25;
  Domain<1>         dom(size);
  T*	real = new T[size];
  T*	imag = new T[size];

  for (index_type i=0; i<size; ++i)
  {
    real[i] = T(3*i+0);
    imag[i] = T(3*i+2);
  }

  {

    Dense<1, complex<T> > block(dom, real, imag);

    insist(block.size()         == size);
    insist(block.admitted()     == false);
    insist(block.user_storage() == split_format);
    
    block.admit(true);
    
    insist(block.admitted()     == true);
    
    for (index_type i=0; i<size; ++i)
      insist(equal(block.get(i), complex<T>(T(3*i+0), T(3*i+2))) );
    
    for (index_type i=0; i<size; ++i)
      block.put(i, complex<T>(T(2*i+1), T(3*i+1)));
    
    block.release(true);
    
    insist(block.admitted()     == false);
  }
    
  for (index_type i=0; i<size; ++i)
  {
    insist(equal(real[i], T(2*i+1)));
    insist(equal(imag[i], T(3*i+1)));
  }

  delete[] real;
  delete[] imag;
}



template <typename T>
void
test_1_interleaved()
{
  length_type const size = 25;
  Domain<1>         dom(size);
  T*	data = new T[2*size];

  for (index_type i=0; i<size; ++i)
  {
    data[2*i+0] = T(3*i+0);
    data[2*i+1] = T(3*i+2);
  }

  {
    Dense<1, complex<T> > block(dom, data);

    insist(block.size()         == size);
    insist(block.admitted()     == false);
    insist(block.user_storage() == interleaved_format);
    
    block.admit(true);

    insist(block.admitted()     == true);
      
    for (index_type i=0; i<size; ++i)
      insist(equal(block.get(i), complex<T>(T(3*i+0), T(3*i+2))) );

    for (index_type i=0; i<size; ++i)
      block.put(i, complex<T>(T(2*i+1), T(3*i+1)));

    block.release(true);

    insist(block.admitted()     == false);
  }

  for (index_type i=0; i<size; ++i)
  {
    insist(equal(data[2*i+0], T(2*i+1)));
    insist(equal(data[2*i+1], T(3*i+1)));
  }

  delete[] data;
}


template <typename T,
	  typename Order>
void
test_2_scalar()
{
  length_type const num_rows = 15;
  length_type const num_cols = 25;
  Domain<2>         dom(num_rows, num_cols);
  length_type const size = num_rows*num_cols;

  bool const is_row_major = is_same<Order, vsip::row2_type>::value;
  assert((is_same<Order, vsip::row2_type>::value ||
	  is_same<Order, vsip::col2_type>::value));

  T*	data = new T[size];

  for (index_type i=0; i<num_rows*num_cols; ++i)
    data[i] = T(i);

  {
    Dense<2, T, Order> block(dom, data);

    insist(block.size()         == num_rows*num_cols);
    insist(block.size(1, 0)     == num_rows*num_cols);
    insist(block.size(2, 0)     == num_rows);
    insist(block.size(2, 1)     == num_cols);
    insist(block.admitted()     == false);
    insist(block.user_storage() == array_format);

    block.admit(true);

    insist(block.admitted()     == true);
      
    for (index_type r=0; r<num_rows; ++r)
    {
      for (index_type c=0; c<num_cols; ++c)
      {
	T val = is_row_major ? T(r*num_cols+c) : T(r+c*num_rows);
	insist(equal(block.get(r, c), val));
      }
    }

    for (index_type r=0; r<num_rows; ++r)
      for (index_type c=0; c<num_cols; ++c)
	block.put(r, c, T(r*num_cols+c+1));

    block.release(true);

    insist(block.admitted()     == false);
  }

  for (index_type i=0; i<size; ++i)
  {
    index_type r = is_row_major ? i / num_cols : i % num_rows;
    index_type c = is_row_major ? i % num_cols : i / num_rows;

    insist(equal(data[i], T(r*num_cols+c+1)));
  }

  delete[] data;
}



template <typename T,
	  typename Order>
void
test_2_split()
{
  length_type const num_rows = 15;
  length_type const num_cols = 25;
  Domain<2>         dom(num_rows, num_cols);
  length_type const size = num_rows*num_cols;

  bool const is_row_major = is_same<Order, vsip::row2_type>::value;
  assert((is_same<Order, vsip::row2_type>::value ||
	  is_same<Order, vsip::col2_type>::value));

  T*	real = new T[size];
  T*	imag = new T[size];

  for (index_type i=0; i<num_rows*num_cols; ++i)
  {
    real[i] = T(2*i+0);
    imag[i] = T(2*i+1);
  }

  {
    Dense<2, complex<T>, Order> block(dom, real, imag);

    insist(block.size()         == num_rows*num_cols);
    insist(block.size(1, 0)     == num_rows*num_cols);
    insist(block.size(2, 0)     == num_rows);
    insist(block.size(2, 1)     == num_cols);
    insist(block.admitted()     == false);
    insist(block.user_storage() == split_format);

    block.admit(true);

    insist(block.admitted()     == true);
      
    for (index_type r=0; r<num_rows; ++r)
    {
      for (index_type c=0; c<num_cols; ++c)
      {
	index_type i = is_row_major ? r*num_cols+c : r+c*num_rows;
	complex<T> val = complex<T>(2*i+0, 2*i+1);
	insist(equal(block.get(r, c), val));
      }
    }

    for (index_type r=0; r<num_rows; ++r)
      for (index_type c=0; c<num_cols; ++c)
	block.put(r, c, complex<T>(r*num_cols+c,r+c*num_rows));

    block.release(true);

    insist(block.admitted()     == false);
  }

  for (index_type i=0; i<size; ++i)
  {
    index_type r = is_row_major ? i / num_cols : i % num_rows;
    index_type c = is_row_major ? i % num_cols : i / num_rows;

    insist(equal(real[i], T(r*num_cols+c)));
    insist(equal(imag[i], T(r+c*num_rows)));
  }

  delete[] real;
  delete[] imag;
}



template <typename T,
	  typename Order>
void
test_2_interleaved()
{
  length_type const num_rows = 15;
  length_type const num_cols = 25;
  length_type const size = num_rows*num_cols;
  Domain<2>         dom(num_rows, num_cols);

  bool const is_row_major = is_same<Order, vsip::row2_type>::value;
  assert((is_same<Order, vsip::row2_type>::value ||
	  is_same<Order, vsip::col2_type>::value));

  T*	data = new T[2*size];

  for (index_type i=0; i<num_rows*num_cols; ++i)
  {
    data[2*i+0] = T(2*i+0);
    data[2*i+1] = T(2*i+1);
  }

  {
    Dense<2, complex<T>, Order> block(dom, data);

    insist(block.size()         == num_rows*num_cols);
    insist(block.size(1, 0)     == num_rows*num_cols);
    insist(block.size(2, 0)     == num_rows);
    insist(block.size(2, 1)     == num_cols);
    insist(block.admitted()     == false);
    insist(block.user_storage() == interleaved_format);

    block.admit(true);

    insist(block.admitted()     == true);
      
    for (index_type r=0; r<num_rows; ++r)
    {
      for (index_type c=0; c<num_cols; ++c)
      {
	index_type i = is_row_major ? r*num_cols+c : r+c*num_rows;
	complex<T> val = complex<T>(2*i+0, 2*i+1);
	insist(equal(block.get(r, c), val));
      }
    }

    for (index_type r=0; r<num_rows; ++r)
      for (index_type c=0; c<num_cols; ++c)
	block.put(r, c, complex<T>(r*num_cols+c,r+c*num_rows));

    block.release(true);

    insist(block.admitted()     == false);
  }

  for (index_type i=0; i<size; ++i)
  {
    index_type r = is_row_major ? i / num_cols : i % num_rows;
    index_type c = is_row_major ? i % num_cols : i / num_rows;

    insist(equal(data[2*i+0], T(r*num_cols+c)));
    insist(equal(data[2*i+1], T(r+c*num_rows)));
  }

  delete[] data;
}



int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);

  test_1_scalar<scalar_f>();
  test_1_scalar<scalar_i>();
  // test_1_scalar<index_type>(); // impl by tvcpp
  // test_1_scalar<vsip::Index<1> >(); // impl by tvcpp

  test_1_split<scalar_f>();
  // test_1_split<scalar_i>(); // not impl by tvcpp

  test_1_interleaved<scalar_f>();

  test_2_scalar<scalar_f, vsip::row2_type>();
  test_2_scalar<scalar_f, vsip::col2_type>();
  test_2_scalar<scalar_i, vsip::row2_type>();
  test_2_scalar<scalar_i, vsip::col2_type>();

  test_2_split<scalar_f, vsip::row2_type>();
  test_2_split<scalar_f, vsip::col2_type>();

  test_2_interleaved<scalar_f, vsip::row2_type>();
  test_2_interleaved<scalar_f, vsip::col2_type>();

  return EXIT_SUCCESS;
}

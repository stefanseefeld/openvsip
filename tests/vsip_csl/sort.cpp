/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/sort.cpp
    @author  Mike LeBlanc
    @date    2009-04-22
    @brief   VSIPL++ Library: Unit tests for Sort
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip_csl/sort.hpp>
#include <vsip_csl/test.hpp>

#include "test-random.hpp"

using namespace std;
using namespace vsip;
using namespace vsip_csl;


template <typename T,
	  typename Block>
inline
std::ostream&
operator<<(
  std::ostream&		       out,
  vsip::const_Vector<T, Block> vec)
  VSIP_NOTHROW
{
  for (vsip::index_type i=0; i<vec.size(); ++i)
    out << "  " << i << ": " << vec.get(i) << "\n";
  return out;
}

/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

// fill_vector -- fill a vector with a sequence of values.
//
// Values are generated with a slope of K.

template <typename T,
	  typename Block>
void
fill_vector(Vector<T, Block> vec, int k)
{
  for (index_type i=0; i<vec.size(0); ++i)
    vec.put(i, T(k*i));
}


// check_sorted_indices -- check that an index vector has been
// permuted properly for the values in a data vector.

template <
          typename T,
	  typename FunctorT
         >
bool
check_sorted_indices(
  Vector<index_type> indx,
  Vector<T>  data,
  FunctorT   comp
  )
{
  length_type n = data.size(0);
  for( index_type i = 0; i<(n-1); ++i )
    if( comp(data.get(indx.get(i+1)), data.get(indx.get(i))) )
      return false;
  return true;
}

// check_sorted_indices -- apply above with less<T>.

template <typename T>
bool
check_sorted_indices(
  Vector<index_type> indx,
  Vector<T>  data
  )
{
  return check_sorted_indices(indx,data,less<T>());
}

// check_sorted_data: check that a data vector is
// sorted properly.

template <
          typename T,
	  typename FunctorT
         >
bool
check_sorted_data(
  Vector<T>  data,
  FunctorT   comp
  )
{
  length_type n = data.size(0);
  for(index_type i = 0; i<(n-1); ++i )
    if( comp(data.get(i+1), data.get(i)) )
      return false;
  return true;
}

// check_sorted_data: apply above with less<T>

template <typename T>
bool
check_sorted_data(
  Vector<T>  data
  )
{
  return check_sorted_data(data,less<T>());
}

// Specialize randv<int> so we don't get a vector
// full of zeroes.

template <>
void
randv<int>(vsip::Vector<int> v)
{
  vsip::Rand<float> rgen(1, true);

  v = rgen.randu(v.size()) * v.size();
}

// test_sort_indices:
//   Test sort_indices() with calls containing both
//   input (data) and output (index) views.
  
template <typename T>
void
test_sort_indices(length_type n)
{
  Vector<T>          data(n);
  Vector<T>          copy(n);
  Vector<index_type> indx(n);
  Vector<index_type> indx2(n);

  randv(data);
  copy = data;

  fill_vector(indx,1);
  sort_indices(indx, data);
  // Check that we permuted the indices properly.
  test_assert(check_sorted_indices(indx,data));
  // Check that we did NOT change the data.
  test_assert(view_equal(data,copy));

  // Same as above but using explicitly the functor
  // we think is the default.  Check that we produce
  // the same permutation.
  indx2 = indx;
  fill_vector(indx,1);
  sort_indices(indx, data, less<T>());
  test_assert(check_sorted_indices(indx,data));
  test_assert(view_equal(data,copy));
  test_assert(view_equal(indx,indx2));

  // Check that we can sort in descending sorder.
  fill_vector(indx,1);
  sort_indices(indx, data, greater<T>());
  test_assert(check_sorted_indices(indx,data,greater<T>()));
  test_assert(view_equal(data,copy));
}

// test_sort_data:
//   Test sort_data() with calls containing both
//   input and output views.

template <typename T>
void
test_sort_data(length_type n)
{
  Vector<T>          data(n);
  Vector<T>          data2(n);
  Vector<T>          copy(n);

  randv(data);
  copy = data;

  sort_data(data, data2);
  test_assert(check_sorted_data(data2));
  // Check that we did NOT change the input data.
  test_assert(view_equal(data,copy));

  // Same as above but using explicitly the functor
  // we think is the default.  Check that we produce
  // the same permutation.
  Vector<T>          data3(n);
  data3 = data2;
  sort_data(data, data2, less<T>());
  test_assert(check_sorted_data(data2));
  test_assert(view_equal(data,copy));
  test_assert(view_equal(data2,data3));

  // Check that we can sort in descending sorder.
  sort_data(data, data2, greater<T>());
  test_assert(check_sorted_data(data2,greater<T>()));
  test_assert(view_equal(data,copy));
}

// test_sort_data_ip:
//   Test sort_data_ip() with calls containing
//   an input / output view.

template <typename T>
void
test_sort_data_ip(length_type n)
{
  Vector<T>          data(n);
  Vector<T>          copy(n);

  randv(data);
  copy = data;

  sort_data(data);
  test_assert(check_sorted_data(data));

  // Same as above but using explicitly the functor
  // we think is the default.  Check that we produce
  // the same permutation.
  Vector<T>          data3(n);
  data3 = data;
  data = copy;
  sort_data(data, less<T>());
  test_assert(check_sorted_data(data));
  test_assert(view_equal(data,data3));

  // Check that we can sort in descending sorder.
  data = copy;
  sort_data(data, greater<T>());
  test_assert(check_sorted_data(data,greater<T>()));
}


/***********************************************************************
  Definitions - Sort Test Cases.
***********************************************************************/


int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  const length_type N = 10;

  test_sort_indices<int>(N);
  test_sort_indices<float>(N);
  test_sort_indices<double>(N);

  test_sort_data<int>(N);
  test_sort_data<float>(N);
  test_sort_data<double>(N);

  test_sort_data_ip<int>(N);
  test_sort_data_ip<float>(N);
  test_sort_data_ip<double>(N);

  return 0;
}

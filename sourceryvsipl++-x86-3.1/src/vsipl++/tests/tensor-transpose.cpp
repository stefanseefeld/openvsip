/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/tensor-transpose.cpp
    @author  Jules Bergmann
    @date    2005-08-18
    @brief   VSIPL++ Library: Unit tests for tensor transpose subviews.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>

#include <vsip_csl/test.hpp>

#define USE_TRANSPOSE_VIEW_TYPEDEF 1

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

// Fill tensor values with ramp.

template <typename TensorT>
void
fill_tensor(TensorT view, int offset=0)
{
  typedef typename TensorT::value_type T;

  length_type const size1 = view.size(1);
  length_type const size2 = view.size(2);

  // Initialize view

  for (index_type idx0=0; idx0<view.size(0); ++idx0)
    for (index_type idx1=0; idx1<view.size(1); ++idx1)
      for (index_type idx2=0; idx2<view.size(2); ++idx2)
	view.put(idx0, idx1, idx2, T(idx0 * size1 * size2 +
				     idx1 * size2         +
				     idx2 + offset));
}



// Check tensor values against ramp.

template <typename TensorT>
void
check_tensor(TensorT view, int offset=0)
{
  typedef typename TensorT::value_type T;

  length_type const size1 = view.size(1);
  length_type const size2 = view.size(2);

  for (index_type idx0=0; idx0<view.size(0); ++idx0)
    for (index_type idx1=0; idx1<view.size(1); ++idx1)
      for (index_type idx2=0; idx2<view.size(2); ++idx2)
      {
	test_assert(equal(view.get(idx0, idx1, idx2),
		     T(idx0 * size1 * size2 +
		       idx1 * size2         +
		       idx2 + offset)));
      }
}



template <dimension_type D0,
	  dimension_type D1,
	  dimension_type D2,
	  typename       TensorT>
void
test_transpose_readonly(TensorT view)
{
  typedef typename TensorT::value_type T;

  // Check that view is initialized
  check_tensor(view, 0);

  length_type const size1 = view.size(1);
  length_type const size2 = view.size(2);

#if USE_TRANSPOSE_VIEW_TYPEDEF
  typedef typename TensorT::template transpose_view<D0, D1, D2>::type trans_t;
  trans_t trans = view.template transpose<D0, D1, D2>();
#else
  // ICC-9.0 does not like initial 'typename'.
  typename TensorT::template transpose_view<D0, D1, D2>::type trans =
    view.template transpose<D0, D1, D2>();
#endif

  test_assert(trans.size(0) == view.size(D0));
  test_assert(trans.size(1) == view.size(D1));
  test_assert(trans.size(2) == view.size(D2));

  // Build a reverse dimension map
  dimension_type R0, R1, R2;

  if      (D0 == 0) R0 = 0;
  else if (D0 == 1) R1 = 0;
  else if (D0 == 2) R2 = 0;

  if      (D1 == 0) R0 = 1;
  else if (D1 == 1) R1 = 1;
  else if (D1 == 2) R2 = 1;

  if      (D2 == 0) R0 = 2;
  else if (D2 == 1) R1 = 2;
  else if (D2 == 2) R2 = 2;

  // Sanity check reverse map
  test_assert(trans.size(R0) == view.size(0));
  test_assert(trans.size(R1) == view.size(1));
  test_assert(trans.size(R2) == view.size(2));

  index_type idx[3];

  for (idx[0]=0; idx[0]<trans.size(0); ++idx[0])
    for (idx[1]=0; idx[1]<trans.size(1); ++idx[1])
      for (idx[2]=0; idx[2]<trans.size(2); ++idx[2])
      {
	T expected = T(idx[R0] * size1 * size2 +
		       idx[R1] * size2         +
		       idx[R2]);
	test_assert(equal(trans.get(idx[0], idx[1], idx[2]),
		     expected));
	test_assert(equal(trans.get(idx[0],  idx[1],  idx[2]),
		     view. get(idx[R0], idx[R1], idx[R2])));
      }

  // Check that view is unchanged
  check_tensor(view, 0);
}



template <dimension_type D0,
	  dimension_type D1,
	  dimension_type D2,
	  typename       TensorT>
void
test_transpose(TensorT view)
{
  typedef typename TensorT::value_type T;

  // Check that view is initialized
  check_tensor(view, 0);

  length_type const size1 = view.size(1);
  length_type const size2 = view.size(2);

#if USE_TRANSPOSE_VIEW_TYPEDEF
  typedef typename TensorT::template transpose_view<D0, D1, D2>::type trans_t;
  trans_t trans = view.template transpose<D0, D1, D2>();
#else
  // ICC-9.0 does not like initial 'typename'.
  typename TensorT::template transpose_view<D0, D1, D2>::type trans =
    view.template transpose<D0, D1, D2>();
#endif

  test_assert(trans.size(0) == view.size(D0));
  test_assert(trans.size(1) == view.size(D1));
  test_assert(trans.size(2) == view.size(D2));

  // Build a reverse dimension map
  dimension_type R0, R1, R2;

  if      (D0 == 0) R0 = 0;
  else if (D0 == 1) R1 = 0;
  else if (D0 == 2) R2 = 0;

  if      (D1 == 0) R0 = 1;
  else if (D1 == 1) R1 = 1;
  else if (D1 == 2) R2 = 1;

  if      (D2 == 0) R0 = 2;
  else if (D2 == 1) R1 = 2;
  else if (D2 == 2) R2 = 2;

  // Sanity check reverse map
  test_assert(trans.size(R0) == view.size(0));
  test_assert(trans.size(R1) == view.size(1));
  test_assert(trans.size(R2) == view.size(2));

  index_type idx[3];

  for (idx[0]=0; idx[0]<trans.size(0); ++idx[0])
    for (idx[1]=0; idx[1]<trans.size(1); ++idx[1])
      for (idx[2]=0; idx[2]<trans.size(2); ++idx[2])
      {
	T expected = T(idx[R0] * size1 * size2 +
		       idx[R1] * size2         +
		       idx[R2]);
	test_assert(equal(trans.get(idx[0], idx[1], idx[2]),
		     expected));
	test_assert(equal(trans.get(idx[0],  idx[1],  idx[2]),
		     view. get(idx[R0], idx[R1], idx[R2])));

	T new_value = T(idx[R0] * size1 * size2 +
			idx[R1] * size2         +
			idx[R2] + 1);
	trans.put(idx[0], idx[1], idx[2], new_value);

	test_assert(equal(trans.get(idx[0],  idx[1],  idx[2]),
		     view. get(idx[R0], idx[R1], idx[R2])));
      }

  // Check that view is changed
  check_tensor(view, 1);
}


template <typename T,
	  typename OrderT>
void
transpose_cases(length_type len0, length_type len1, length_type len2)
{
  Tensor<T, Dense<3, T, OrderT> > view(len0, len1, len2);

  fill_tensor(view); test_transpose<0, 1, 2>(view);
  fill_tensor(view); test_transpose<0, 2, 1>(view);
  fill_tensor(view); test_transpose<1, 0, 2>(view);
  fill_tensor(view); test_transpose<1, 2, 0>(view);
  fill_tensor(view); test_transpose<2, 0, 1>(view);
  fill_tensor(view); test_transpose<2, 1, 0>(view);

  fill_tensor(view);
  const_Tensor<T, Dense<3, T, OrderT> > const_view(view);
  test_transpose_readonly<0, 1, 2>(const_view);
  test_transpose_readonly<0, 2, 1>(const_view);
  test_transpose_readonly<1, 0, 2>(const_view);
  test_transpose_readonly<1, 2, 0>(const_view);
  test_transpose_readonly<2, 0, 1>(const_view);
  test_transpose_readonly<2, 1, 0>(const_view);
  check_tensor(view);
}


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  transpose_cases<float, tuple<0,1,2> >(5, 7, 9);
}

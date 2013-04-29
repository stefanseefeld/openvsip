//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;


/***********************************************************************
  Definitions
***********************************************************************/

// Fill matrix values with ramp.

template <typename MatrixT>
void
fill_matrix(MatrixT view, int offset=0)
{
  typedef typename MatrixT::value_type T;

  length_type const size1 = view.size(1);

  // Initialize view

  for (index_type idx0=0; idx0<view.size(0); ++idx0)
    for (index_type idx1=0; idx1<view.size(1); ++idx1)
      view.put(idx0, idx1, T(idx0 * size1 + idx1 + offset));
}



// Check matrix values against ramp.

template <typename MatrixT>
void
check_matrix(MatrixT view, int offset=0)
{
  typedef typename MatrixT::value_type T;

  length_type const size1 = view.size(1);

  for (index_type idx0=0; idx0<view.size(0); ++idx0)
    for (index_type idx1=0; idx1<view.size(1); ++idx1)
    {
      test_assert(equal(view.get(idx0, idx1),
		   T(idx0 * size1 + idx1 + offset)));
    }
}



template <typename       MatrixT>
void
test_transpose_readonly(MatrixT view)
{
  typedef typename MatrixT::value_type T;

  // Check that view is initialized
  check_matrix(view, 0);

  length_type const size1 = view.size(1);

  typename MatrixT::const_transpose_type trans = view.transpose();

  test_assert(trans.size(0) == view.size(1));
  test_assert(trans.size(1) == view.size(0));

  for (index_type idx0=0; idx0<trans.size(0); ++idx0)
    for (index_type idx1=0; idx1<trans.size(1); ++idx1)
    {
      T expected = T(idx1 * size1 + idx0 + 0);
      test_assert(equal(trans.get(idx0, idx1), expected));
      test_assert(equal(trans.get(idx0,  idx1),
		   view. get(idx1, idx0)));
      }

  // Check that view is unchanged
  check_matrix(view, 0);
}



template <typename       MatrixT>
void
test_transpose(MatrixT view)
{
  typedef typename MatrixT::value_type T;

  // Check that view is initialized
  check_matrix(view, 0);

  length_type const size1 = view.size(1);

  typename MatrixT::transpose_type trans = view.transpose();

  // Test assignment (forces evaluation, causing the entire
  // view to be computed, possibly by an optimized backend).
  MatrixT tview(trans);

  test_assert(trans.size(0) == view.size(1));
  test_assert(trans.size(1) == view.size(0));

  for (index_type idx0=0; idx0<trans.size(0); ++idx0)
    for (index_type idx1=0; idx1<trans.size(1); ++idx1)
    {
      T expected = T(idx1 * size1 + idx0 + 0);
      test_assert(equal(tview.get(idx0, idx1), expected));
      test_assert(equal(trans.get(idx0, idx1), expected));
      test_assert(equal(trans.get(idx0,  idx1),
		   view. get(idx1, idx0)));

      T new_value = T(idx1 * size1 + idx0 + 1);
      trans.put(idx0, idx1, new_value);

      test_assert(equal(trans.get(idx0,  idx1),
		   view. get(idx1, idx0)));
      }

  // Check that view is changed
  check_matrix(view, 1);
}


template <typename T,
	  typename OrderT>
void
transpose_cases(length_type len0, length_type len1)
{
  Matrix<T, Dense<2, T, OrderT> > view(len0, len1);

  fill_matrix(view); test_transpose(view);

  fill_matrix(view);
  const_Matrix<T, Dense<2, T, OrderT> > const_view(view);
  test_transpose_readonly(const_view);
  check_matrix(view);
}


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  transpose_cases<float, row2_type>(5, 7);
  transpose_cases<float, col2_type>(5, 7);

  transpose_cases<std::complex<float>, row2_type>(5, 7);
  transpose_cases<std::complex<float>, col2_type>(5, 7);

  // Larger, aligned cases which may use an optimized backed
  transpose_cases<float, row2_type>(64, 128);
  transpose_cases<float, col2_type>(128, 64);

  transpose_cases<std::complex<float>, row2_type>(128, 256);
  transpose_cases<std::complex<float>, col2_type>(256, 128);
}

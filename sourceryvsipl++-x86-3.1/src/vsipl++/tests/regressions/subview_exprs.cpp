/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/regr_subview_exprs.cpp
    @author  Jules Bergmann
    @date    2005-09-13
    @brief   VSIPL++ Library: Regression tests expressions with subviews.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 0

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
  Definitions
***********************************************************************/

/// Test A1: Use of subview in assignment.

template <typename T>
void
test_a1(length_type m, length_type n)
{
  Matrix<T> src(m, n, T(1));
  Matrix<T> dst(m, n, T(0));

  for (index_type i=0; i<m; ++i)
    dst.row(i) = src.row(i);

  for (index_type i=0; i<m; ++i)
    for (index_type j=0; j<n; ++j)
      test_assert(equal(dst(i, j), T(1)));
}



/// Test A2: Use of subview in unary element-wise operator.

template <typename T>
void
test_a2(length_type m, length_type n)
{
  Matrix<T> src(m, n, T(-1));
  Matrix<T> dst(m, n, T(0));

  for (index_type i=0; i<m; ++i)
    dst.row(i) = -src.row(i);

  for (index_type i=0; i<m; ++i)
    for (index_type j=0; j<n; ++j)
    {
#if VERBOSE
      if (!(equal(dst(i, j), T(1))))
      {
	std::cout << "test_a2: miscompare at (" << i << ", " << j << "):\n"
		  << "  expected: " << T(1) << "\n"
		  << "  got     : " << dst(i, j) << "\n";
      }
#endif
      test_assert(equal(dst(i, j), T(1)));
    }
}



/// Test A3: Use of subview in unary element-wise function neg.

template <typename T>
void
test_a3(length_type m, length_type n)
{
  Matrix<T> src(m, n, T(-1));
  Matrix<T> dst(m, n, T(0));

  for (index_type i=0; i<m; ++i)
    dst.row(i) = neg(src.row(i));

  for (index_type i=0; i<m; ++i)
    for (index_type j=0; j<n; ++j)
      test_assert(equal(dst(i, j), T(1)));
}



/// Test A4: Use of subview in unary element-wise function mag.

template <typename T>
void
test_a4(length_type m, length_type n)
{
  Matrix<T> src(m, n, T(-1));
  Matrix<T> dst(m, n, T(0));

  for (index_type i=0; i<m; ++i)
    dst.row(i) = mag(src.row(i));

  for (index_type i=0; i<m; ++i)
    for (index_type j=0; j<n; ++j)
      test_assert(equal(dst(i, j), T(1)));
}



/// Test A5: Use of subview in unary element-wise function neg.

template <typename T>
void
test_a5(length_type m, length_type n)
{
  Matrix<T> src(m, n, T(-1));
  Matrix<T> dst(m, n, T(0));

  for (index_type i=0; i<m; ++i)
  {
    typename Matrix<T>::row_type row = src.row(i);
    dst.row(i) = neg(row);
  }

  for (index_type i=0; i<m; ++i)
    for (index_type j=0; j<n; ++j)
      test_assert(equal(dst(i, j), T(1)));
}



/// Test B1:

template <typename T>
void
test_b1(length_type m)
{
  Vector<T> src(m, T(-1));
  Vector<T> dst(m, T(0));

  dst = mag(src);

  for (index_type i=0; i<m; ++i)
    test_assert(equal(dst(i), T(1)));
}



/// Test B2:

template <typename T>
void
test_b2(length_type m)
{
  Vector<T> src(m, T(1));
  Vector<T> dst(m, T(0));

  dst = mag(-src);

  for (index_type i=0; i<m; ++i)
    test_assert(equal(dst(i), T(1)));
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // tests using expressions of row/col subviews
  test_a1<float>(7, 5);	// PASS: dst.row(i) = src.row(i)
  test_a2<float>(7, 5);	// PASS: dst.row(i) = -src.row(i)
  test_a3<float>(7, 5);	// SEGV: dst.row(i) = neg(src.row(i))
  test_a4<float>(7, 5);	// SEGV: dst.row(i) = mag(src.row(i))
  test_a5<float>(7, 5);	// SEGV: row = src.row(i); dst.row(i) = neg(row)

  // tests using expressions of views
  test_b1<float>(7);	// PASS: dst = mag(src)
  test_b2<float>(7);	// PASS: dst = mag(-src)
}

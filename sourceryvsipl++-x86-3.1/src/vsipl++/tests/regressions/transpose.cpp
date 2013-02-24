/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/regr_transpose.cpp
    @author  Jules Bergmann
    @date    2006-02-24
    @brief   VSIPL++ Library: Regression tests for fast transpose.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

template <typename T>
void
test_assign(Domain<2> const& dom)
{
  length_type const rows = dom[0].size();
  length_type const cols = dom[1].size();

  Matrix<T, Dense<2, T, row2_type> > src(rows, cols);
  Matrix<T, Dense<2, T, col2_type> > dst(rows, cols, T());

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
      src(r, c) = T(r*cols+c);

  dst = src;

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
    {
      test_assert(equal(dst(r, c), T(r*cols+c)));
      test_assert(equal(dst(r, c), src(r, c)));
    }
}


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // This test triggers an optimization bug in GreenHills
  // when tail-recursion optimization is not disabled.
  test_assign<complex<float> >(Domain<2>(64, 256)); // segv

  test_assign<float>(Domain<2>(64, 256)); // segv
  test_assign<float>(Domain<2>(16, 16)); // OK
  test_assign<float>(Domain<2>(32, 16)); // OK
  test_assign<float>(Domain<2>(32, 32)); // OK
  test_assign<float>(Domain<2>(64, 64)); // SEGV
}

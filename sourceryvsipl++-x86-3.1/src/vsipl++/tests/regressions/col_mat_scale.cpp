/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/col_mat_scale.cpp
    @author  Jules Bergmann
    @date    2006-09-11
    @brief   VSIPL++ Library: Matrix Scale

    Regression test: SIMD Loop Fusion did not adjust dimension for
      re-dimensioned blocks.  As a result, column-major matrix scale
      failed.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/



template <typename T,
	  typename OrderT>
void
matrix_scale()
{
  typedef Dense<2, T, OrderT> block_type;

  Matrix<T, block_type> view(2, 2);

  view.put(0, 0, T(1, 2));
  view.put(0, 1, T(3, 4));
  view.put(1, 0, T(5, 6));
  view.put(1, 1, T(7, 8));

  view *= T(0.25);

  test_assert(view.get(0, 0) == T(0.25*1, 0.25*2));
  test_assert(view.get(0, 1) == T(0.25*3, 0.25*4));
  test_assert(view.get(1, 0) == T(0.25*5, 0.25*6));
  test_assert(view.get(1, 1) == T(0.25*7, 0.25*8));
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  matrix_scale<complex<float>, row2_type>();
  matrix_scale<complex<float>, col2_type>();

  return 0;
}


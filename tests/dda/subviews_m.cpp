/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/extdata_subviews_m.cpp
    @author  Jules Bergmann
    @date    2005-07-22
    @brief   VSIPL++ Library: Unit tests for DDI to matrix subviews.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <cassert>
#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>

#include <vsip_csl/test.hpp>
#include "output.hpp"
#include "subviews.hpp"

using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

template <typename T,
	  typename OrderT>
void
matrix_test(Domain<2> const& dom)
{
  typedef Dense<2, T, OrderT>           block_type;
  typedef Matrix<T, block_type>         view_type;
  view_type mat(dom[0].size(), dom[1].size());

  test_matrix(mat);
}


template <typename T>
void
test_for_type()
{
  matrix_test<T, row2_type>(Domain<2>(5, 7));
  matrix_test<T, col2_type>(Domain<2>(5, 7));
}




int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  test_for_type<float>();
#else
  test_for_type<float>();
  test_for_type<complex<float> >();
#endif

  return 0;
}

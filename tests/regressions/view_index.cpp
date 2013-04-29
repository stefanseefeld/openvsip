/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/regr_view_index.cpp
    @author  Jules Bergmann
    @date    2005-07-14
    @brief   VSIPL++ Library: Regression test fors views of index_types.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
void
test_vector()
{
  Vector<T> vec(5, T());

  for (index_type i=0; i<vec.size(); ++i)
    test_assert(equal(vec(i), T()));
}



template <typename T>
void
test_matrix()
{
  Matrix<T> mat(5, 7, T());

  for (index_type i=0; i<mat.size(0); ++i)
    for (index_type j=0; j<mat.size(1); ++j)
      test_assert(equal(mat(i, j), T()));
}



template <typename T>
void
test_tensor()
{
  Tensor<T> ten(3, 5, 7, T());

  for (index_type i=0; i<ten.size(0); ++i)
    for (index_type j=0; j<ten.size(1); ++j)
      for (index_type k=0; k<ten.size(2); ++k)
	test_assert(equal(ten(i, j, k), T()));
}



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_vector<float>();		// OK
  test_vector<index_type>();	// Does not compile

  test_matrix<float>();		// OK
  test_matrix<index_type>();	// Does not compile

  test_tensor<float>();		// OK
  test_tensor<index_type>();	// OK
}

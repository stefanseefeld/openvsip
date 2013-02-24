/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/coverage_binary_sub.hpp
    @author  Jules Bergmann
    @date    2005-09-13
    @brief   VSIPL++ Library: Coverage tests for sub binary expressions.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>
#include "common.hpp"

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

TEST_BINARY_OP(sub,  -,  -,  anyval)



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  vector_cases3<Test_sub,  float,           float>();
#else

  vector_cases3<Test_sub, int,             int>();
  vector_cases3<Test_sub, float,           float>();
  vector_cases3<Test_sub, complex<float>,  complex<float> >();
  vector_cases3<Test_sub, complex<float>,  float>();
#if VSIP_IMPL_TEST_DOUBLE
  vector_cases3<Test_sub, double,          double>();
  vector_cases3<Test_sub, complex<double>, complex<double> >();
#endif

  matrix_cases3<Test_sub, float,          float>();
  matrix_cases3<Test_sub, complex<float>, complex<float> >();
#endif

  return EXIT_SUCCESS;
}

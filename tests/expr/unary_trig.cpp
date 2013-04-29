/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/coverage_unary_trid.hpp
    @author  Jules Bergmann
    @date    2005-09-13
    @brief   VSIPL++ Library: Coverage tests for trig unary expressions.
*/

/***********************************************************************
  Included Files
***********************************************************************/

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

TEST_UNARY(sin,   sin,   sin,   anyval)
TEST_UNARY(cos,   cos,   cos,   anyval)
TEST_UNARY(tan,   tan,   tan,   anyval)
TEST_UNARY(atan,  atan,  atan,  anyval)



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0

  vector_cases2<Test_cos, float>();
  vector_cases2<Test_atan, float>();

#else

  // Unary operators
  vector_cases2<Test_cos, float>();
  vector_cases2<Test_cos, complex<float> >();
  vector_cases2<Test_cos, double>();
  vector_cases2<Test_cos, complex<double> >();

  vector_cases2<Test_sin, float>();
  vector_cases2<Test_sin, complex<float> >();
  vector_cases2<Test_sin, double>();
  vector_cases2<Test_sin, complex<double> >();

  vector_cases2<Test_tan, float>();

  vector_cases2<Test_atan, float>();
  vector_cases2<Test_atan, double>();

#endif // VSIP_IMPL_TEST_LEVEL > 0
}

/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/coverage_unary_log.hpp
    @author  Jules Bergmann
    @date    2005-09-13
    @brief   VSIPL++ Library: Coverage tests for logarithm unary expressions.
*/

#include <vsip/initfin.hpp>
#include "common.hpp"

using namespace vsip;

TEST_UNARY(log,   log,   log,   posval)
TEST_UNARY(log10, log10, log10, posval)

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0

  vector_cases2<Test_log, float>();

#else

  // Unary operators
  vector_cases2<Test_log, float>();
  vector_cases2<Test_log, complex<float> >();
  vector_cases2<Test_log, double>();
  vector_cases2<Test_log, complex<double> >();

  vector_cases2<Test_log10, float>();
  vector_cases2<Test_log10, complex<float> >();
  vector_cases2<Test_log10, double>();
  vector_cases2<Test_log10, complex<double> >();

#endif // VSIP_IMPL_TEST_LEVEL > 0
}

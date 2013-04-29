/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/coverage_unary_neg.hpp
    @author  Jules Bergmann
    @date    2005-09-13
    @brief   VSIPL++ Library: Coverage tests for neg unary expressions.
*/

#include <vsip/initfin.hpp>
#include <vsip/opt/assign_diagnostics.hpp>
#include "common.hpp"

using namespace vsip;

TEST_UNARY(neg,   -,     -,     anyval)

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0

  vector_cases2<Test_neg, float>();

#else

  // Unary operators
  vector_cases2<Test_neg, int>();
  vector_cases2<Test_neg, float>();
  vector_cases2<Test_neg, double>();
  vector_cases2<Test_neg, complex<float> >();
  vector_cases2<Test_neg, complex<double> >();

#endif // VSIP_IMPL_TEST_LEVEL > 0
}

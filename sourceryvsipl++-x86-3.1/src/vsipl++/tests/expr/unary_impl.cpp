/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/coverage_unary_impl.hpp
    @author  Jules Bergmann
    @date    2005-09-13
    @brief   VSIPL++ Library: Coverage tests for Sourcery VSIPL++
             implementation specific unary expressions.
*/

#include <vsip/initfin.hpp>
#include "common.hpp"

using namespace vsip;

TEST_UNARY(is_nan,    is_nan,    impl::fn::is_nan,    anyval)
// These C99 functions are unavailable on Windows.
#if !defined(_MSC_VER)
TEST_UNARY(is_finite, is_finite, impl::fn::is_finite, anyval)
TEST_UNARY(is_normal, is_normal, impl::fn::is_normal, anyval)
#endif


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  vector_cases2_rt<Test_is_nan,    float,  bool>();
#if !defined(_MSC_VER)
  vector_cases2_rt<Test_is_finite, float,  bool>();
  vector_cases2_rt<Test_is_normal, float,  bool>();
#endif
}

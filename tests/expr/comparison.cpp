/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/coverage_comparison.hpp
    @author  Jules Bergmann
    @date    2006-06-01
    @brief   VSIPL++ Library: Coverage tests for comparison operations.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>
#include "common.hpp"

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

TEST_BINARY_FUNC(gt, gt, gt, anyval)
TEST_BINARY_FUNC(lt, lt, lt, anyval)



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  vector_cases3_bool<Test_gt, int,   int>();
  vector_cases3_bool<Test_gt, float, float>();
  vector_cases3_bool<Test_lt, int,   int>();
  vector_cases3_bool<Test_lt, float, float>();
}

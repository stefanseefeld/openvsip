//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include "common.hpp"

TEST_BINARY_FUNC(gt, gt, gt, anyval)
TEST_BINARY_FUNC(lt, lt, lt, anyval)

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  vector_cases3_bool<Test_gt, int,   int>();
  vector_cases3_bool<Test_gt, float, float>();
  vector_cases3_bool<Test_lt, int,   int>();
  vector_cases3_bool<Test_lt, float, float>();
}

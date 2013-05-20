//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include "common.hpp"

TEST_BINARY_OP(mul,  *,  *,  anyval)

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  matrix_cases3<Test_mul, float, float>();
  matrix_cases3<Test_mul, complex<float>, complex<float> >();

  return EXIT_SUCCESS;
}

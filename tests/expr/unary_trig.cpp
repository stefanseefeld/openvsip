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

TEST_UNARY(sin,   sin,   sin,   anyval)
TEST_UNARY(cos,   cos,   cos,   anyval)
TEST_UNARY(tan,   tan,   tan,   anyval)
TEST_UNARY(atan,  atan,  atan,  anyval)

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

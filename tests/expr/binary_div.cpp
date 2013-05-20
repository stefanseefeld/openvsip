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

TEST_BINARY_OP(div,  /,  /,  nonzero)

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  vector_cases3<Test_div, int,             int>();
  vector_cases3<Test_div,  complex<float>,  float>();
#else

  vector_cases3<Test_div, int,             int>();
  vector_cases3<Test_div, float,           float>();
  vector_cases3<Test_div, complex<float>,  complex<float> >();
  vector_cases3<Test_div, complex<float>,  float>();
#if VSIP_IMPL_TEST_DOUBLE
  vector_cases3<Test_div, double,          double>();
  vector_cases3<Test_div, complex<double>, complex<double> >();
#endif
#endif
}

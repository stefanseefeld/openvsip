//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

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

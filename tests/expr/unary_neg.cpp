//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

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

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

TEST_BINARY_OP(add,  +,  +,  anyval)

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  vector_cases3<Test_add,  int,             int>();
#else

  // Binary Operators
  vector_cases3<Test_add, int,             int>();
  vector_cases3<Test_add, float,           float>();
  vector_cases3<Test_add, complex<float>,  complex<float> >();
  vector_cases3<Test_add, float,           complex<float> >();
  vector_cases3<Test_add, complex<float>,  float>();
#if VSIP_IMPL_TEST_DOUBLE
  vector_cases3<Test_add, double,          double>();
  vector_cases3<Test_add, complex<double>, complex<double> >();
#endif
#endif // VSIP_IMPL_TEST_LEVEL == 0
}

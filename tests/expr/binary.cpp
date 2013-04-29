//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>

#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>
#include "common.hpp"

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

TEST_BINARY_FUNC(max,  max,  max,  anyval)
TEST_BINARY_FUNC(min,  min,  min,  anyval)
TEST_BINARY_FUNC(band, band, band, anyval)
TEST_BINARY_FUNC(bor,  bor,  bor,  anyval)
TEST_BINARY_FUNC(bxor, bxor, bxor, anyval)
TEST_BINARY_FUNC(land, land, land, anyval)
TEST_BINARY_FUNC(lor,  lor,  lor,  anyval)
TEST_BINARY_FUNC(lxor, lxor, lxor, anyval)



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  vector_cases3<Test_max,  float,  float>();
  vector_cases3<Test_min,  float,  float>();
  vector_cases3<Test_band, int,    int>();
  vector_cases3<Test_lxor, bool,   bool>();
#else

  vector_cases3<Test_max, float,   float>();
  vector_cases3<Test_min, float,   float>();

#if VSIP_IMPL_TEST_DOUBLE
  vector_cases3<Test_max, double,  double>();
  vector_cases3<Test_min, double,  double>();
#endif

  vector_cases3<Test_band, int,    int>();
  vector_cases3<Test_bor,  int,    int>();
  vector_cases3<Test_bxor, int,    int>();

  vector_cases3<Test_land, bool,   bool>();
  vector_cases3<Test_lor,  bool,   bool>();
  vector_cases3<Test_lxor, bool,   bool>();
#endif

  return EXIT_SUCCESS;
}

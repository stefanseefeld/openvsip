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

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>
#include "common.hpp"

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

TEST_UNARY(exp,   exp,   exp,   posval)
TEST_UNARY(exp10, exp10, exp10, posval)



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0

  vector_cases2<Test_exp, float>();

#else

  // Unary operators
  vector_cases2<Test_exp, float>();
  vector_cases2<Test_exp, double>();

  vector_cases2<Test_exp10, float>();
  vector_cases2<Test_exp10, double>();

#endif // VSIP_IMPL_TEST_LEVEL > 0
}

//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

// VERBOSE is recognized by coverage_common.hpp
#define VERBOSE 0

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

TEST_UNARY(copy,   ,      ,     anyval)
TEST_UNARY(mag,   mag,   mag,   anyval)
TEST_UNARY(sqrt,  sqrt,  sqrt,  posval)
TEST_UNARY(rsqrt, rsqrt, rsqrt, posval)
TEST_UNARY(sq,    sq,    sq,    anyval)
TEST_UNARY(recip, recip, recip, nonzero)

TEST_UNARY(bnot,  bnot,  bnot,  anyval)
TEST_UNARY(lnot,  lnot,  lnot,  anyval)
TEST_UNARY(conj,  conj,  conj,  anyval)





/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0

  vector_cases2<Test_mag, float>();
  vector_cases2_rt<Test_mag, complex<float>,  float>();
  vector_cases2<Test_sqrt, float>();
  vector_cases2<Test_sq, float>();
  vector_cases2<Test_copy, float>();
  vector_cases2_mix<Test_copy, complex<float> >();
  vector_cases2<Test_bnot, int>();
  vector_cases2<Test_lnot, bool>();
  vector_cases2<Test_conj, complex<float> >();

#else

  // Unary operators
  vector_cases2<Test_mag, int>();
  vector_cases2<Test_mag, float>();
  vector_cases2<Test_mag, double>();
  vector_cases2_rt<Test_mag, complex<float>,  float>();

  vector_cases2<Test_sqrt, float>();
  vector_cases2<Test_sqrt, double>();
  vector_cases2<Test_sqrt, complex<float> >();

  vector_cases2<Test_rsqrt, float>();
  vector_cases2<Test_rsqrt, double>();

  vector_cases2<Test_sq, float>();
  vector_cases2<Test_sq, double>();

  vector_cases2<Test_recip, float>();
  vector_cases2<Test_recip, complex<float> >();
  vector_cases2<Test_recip, complex<double> >();

  vector_cases2<Test_copy, float>();
  vector_cases2<Test_copy, complex<float> >();
  vector_cases2<Test_copy, complex<double> >();

  vector_cases2_mix<Test_copy, complex<float> >();
  vector_cases2_mix<Test_copy, complex<double> >();

  vector_cases2<Test_bnot, int>();
  vector_cases2<Test_lnot, bool>();

  vector_cases2<Test_conj, complex<float> >();
  vector_cases2<Test_conj, complex<double> >();

#endif // VSIP_IMPL_TEST_LEVEL > 0
}

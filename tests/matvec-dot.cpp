//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <cassert>


#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/math.hpp>

#include <vsip_csl/ref_matvec.hpp>

#include <vsip_csl/test.hpp>
#include "test-random.hpp"
#include <vsip_csl/test-precision.hpp>

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#  include <vsip_csl/output.hpp>
#endif

using namespace std;
using namespace vsip;
namespace ref = vsip_csl::ref;
using vsip_csl::equal;
using vsip_csl::almost_equal;
using vsip_csl::Precision_traits;


/***********************************************************************
  Definitions
***********************************************************************/

/// Test dot product with random values.

template <typename T0,
	  typename T1>
void
test_dot_rand(length_type m)
{
  typedef typename Promotion<T0, T1>::type return_type;
  typedef typename vsip::impl::scalar_of<return_type>::type scalar_type;

  Vector<T0> a(m);
  Vector<T1> b(m);

  randv(a);
  randv(b);

  // Test vector-vector prod
  return_type val = dot(a, b);
  return_type chk = ref::dot(a, b);

  // Verify the result using a slightly lower threshold since 'equal(val, chk)'
  //  fails on CELL builds due to small numerical inconsistencies.
  scalar_type err_threshold = 1e-4;
  test_assert(almost_equal(val, chk, err_threshold));
}



/// Test conjugated vector dot product with random values.

template <typename T0,
	  typename T1>
void
test_cvjdot_rand(length_type m)
{
  typedef typename Promotion<T0, T1>::type return_type;
  typedef typename vsip::impl::scalar_of<return_type>::type scalar_type;

  Vector<T0> a(m);
  Vector<T1> b(m);

  randv(a);
  randv(b);

  // Test vector-vector prod
  return_type val  = cvjdot(a, b);
  return_type chk1 = dot(a, conj(b));
  return_type chk2 = ref::dot(a, conj(b));

  test_assert(equal(val, chk1));
  test_assert(equal(val, chk2));
}



template <typename T0,
	  typename T1>
void
dot_cases()
{
  for (length_type m=16; m<16384; m *= 4)
  {
    test_dot_rand<T0, T1>(m);
    test_dot_rand<T0, T1>(m+1);
    test_dot_rand<T0, T1>(2*m);
  }
}



template <typename T0,
	  typename T1>
void
cvjdot_cases()
{
  for (length_type m=16; m<16384; m *= 4)
  {
    test_cvjdot_rand<T0, T1>(m);
    test_cvjdot_rand<T0, T1>(m+1);
    test_cvjdot_rand<T0, T1>(2*m);
  }
}



void
dot_types()
{
  dot_cases<float,  float>();

  dot_cases<complex<float>, complex<float> >();
  dot_cases<float,          complex<float> >();
  dot_cases<complex<float>, float>();

  cvjdot_cases<complex<float>,  complex<float> >();

#if VSIP_IMPL_TEST_DOUBLE
  dot_cases<float,  double>();
  dot_cases<double, float>();
  dot_cases<double, double>();

  cvjdot_cases<complex<double>, complex<double> >();
#endif
}



/***********************************************************************
  Main
***********************************************************************/

template <> float  Precision_traits<float>::eps = 0.0;
template <> double Precision_traits<double>::eps = 0.0;

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  Precision_traits<float>::compute_eps();
  Precision_traits<double>::compute_eps();

  test_cvjdot_rand<complex<float>, complex<float> >(16);

  dot_types();
}

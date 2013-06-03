//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/math.hpp>
#include <test/ref/matvec.hpp>
#include <test.hpp>

#define VERBOSE 0

using namespace ovxx;

/// Test dot product with random values.

template <typename T0,
	  typename T1>
void
test_dot_rand(length_type m)
{
  typedef typename Promotion<T0, T1>::type return_type;
  typedef typename scalar_of<return_type>::type scalar_type;

  Vector<T0> a(m);
  Vector<T1> b(m);

  test::randv(a);
  test::randv(b);

  // Test vector-vector prod
  return_type val = dot(a, b);
  return_type chk = test::ref::dot(a, b);

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
  typedef typename scalar_of<return_type>::type scalar_type;

  Vector<T0> a(m);
  Vector<T1> b(m);

  test::randv(a);
  test::randv(b);

  // Test vector-vector prod
  return_type val  = cvjdot(a, b);
  return_type chk1 = dot(a, conj(b));
  return_type chk2 = test::ref::dot(a, conj(b));

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

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test::precision<float>::init();
  test::precision<double>::init();

  test_cvjdot_rand<complex<float>, complex<float> >(16);

  dot_types();
}

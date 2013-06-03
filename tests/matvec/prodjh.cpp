//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/math.hpp>
#include <test/ref/matvec.hpp>
#include <test.hpp>
#include "prod.hpp"

using namespace ovxx;

template <typename T0,
	  typename T1>
void
test_prodh_rand(length_type m, length_type n, length_type k)
{
  typedef typename Promotion<T0, T1>::type return_type;
  typedef typename scalar_of<return_type>::type scalar_type;

  Matrix<T0> a(m, n);
  Matrix<T1> b(k, n);
  Matrix<return_type> res1(m, k);
  Matrix<return_type> chk(m, k);
  Matrix<scalar_type> gauge(m, k);

  test::randm(a);
  test::randm(b);

  // Test matrix-matrix prod for hermitian
  res1   = prodh(a, b);

  chk   = test::ref::prod(a, herm(b));
  gauge = test::ref::prod(mag(a), mag(herm(b)));

  for (index_type i=0; i<gauge.size(0); ++i)
    for (index_type j=0; j<gauge.size(1); ++j)
      if (!(gauge(i, j) > scalar_type()))
	gauge(i, j) = scalar_type(1);

#if VERBOSE
  cout << "a     =\n" << a;
  cout << "b     =\n" << b;
#endif

  check_prod( res1, chk, gauge );
}



template <typename T0,
	  typename T1>
void
test_prodj_rand(length_type m, length_type n, length_type k)
{
  typedef typename Promotion<T0, T1>::type return_type;
  typedef typename scalar_of<return_type>::type scalar_type;

  Matrix<T0> a(m, n);
  Matrix<T1> b(n, k);
  Matrix<return_type> res1(m, k);
  Matrix<return_type> chk(m, k);
  Matrix<scalar_type> gauge(m, k);

  test::randm(a);
  test::randm(b);

  // Test matrix-matrix prod for hermitian
  res1   = prodj(a, b);

  chk   = test::ref::prod(a, conj(b));
  gauge = test::ref::prod(mag(a), mag(conj(b)));

  for (index_type i=0; i<gauge.size(0); ++i)
    for (index_type j=0; j<gauge.size(1); ++j)
      if (!(gauge(i, j) > scalar_type()))
	gauge(i, j) = scalar_type(1);

#if VERBOSE
  cout << "a     =\n" << a;
  cout << "b     =\n" << b;
#endif

  check_prod( res1, chk, gauge );
}



template <typename T0,
	  typename T1>
void
prod_cases_complex_only()
{
  test_prodh_rand<T0, T1>(5, 5, 5);
  test_prodh_rand<T0, T1>(5, 7, 9);
  test_prodh_rand<T0, T1>(9, 5, 7);
  test_prodh_rand<T0, T1>(9, 7, 5);

  test_prodj_rand<T0, T1>(5, 5, 5);
  test_prodj_rand<T0, T1>(5, 7, 9);
  test_prodj_rand<T0, T1>(9, 5, 7);
  test_prodj_rand<T0, T1>(9, 7, 5);
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test::precision<float>::init();
  test::precision<double>::init();

  prod_cases_complex_only<complex<float>, complex<float> >();
}

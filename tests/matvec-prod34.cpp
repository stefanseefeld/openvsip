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

#include <vsip_csl/output.hpp>
#include <vsip_csl/ref_matvec.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/test-precision.hpp>

#include "test-prod.hpp"
#include "test-random.hpp"

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Test Definitions
***********************************************************************/

template <typename T0,
	  typename T1>
void
test_prod3_rand()
{
  typedef typename Promotion<T0, T1>::type return_type;
  typedef typename vsip::impl::scalar_of<return_type>::type scalar_type;
  const length_type m = 3;
  const length_type n = 3;
  const length_type k = 3;

  Matrix<T0> a(m, n);
  Matrix<T1> b(n, k);
  Matrix<return_type> res1(m, k);
  Matrix<return_type> res2(m, k);
  Matrix<return_type> chk(m, k);
  Matrix<scalar_type> gauge(m, k);

  randm(a);
  randm(b);

  // Test matrix-matrix prod
  res1   = prod3(a, b);

  // Test matrix-vector prod
  for (index_type i=0; i<k; ++i)
    res2.col(i) = prod3(a, b.col(i));

  chk   = ref::prod(a, b);
  gauge = ref::prod(mag(a), mag(b));

  for (index_type i=0; i<gauge.size(0); ++i)
    for (index_type j=0; j<gauge.size(1); ++j)
      if (!(gauge(i, j) > scalar_type()))
	gauge(i, j) = scalar_type(1);

#if VERBOSE
  cout << "a     =\n" << a;
  cout << "b     =\n" << b;
  cout << "chk   =\n" << chk;
  cout << "res1  =\n" << res1;
  cout << "res2  =\n" << res2;
#endif

  check_prod( res1, chk, gauge );
  check_prod( res2, chk, gauge );
}


template <typename T0,
	  typename T1>
void
test_prod4_rand()
{
  typedef typename Promotion<T0, T1>::type return_type;
  typedef typename vsip::impl::scalar_of<return_type>::type scalar_type;
  const length_type m = 4;
  const length_type n = 4;
  const length_type k = 4;

  Matrix<T0> a(m, n);
  Matrix<T1> b(n, k);
  Matrix<return_type> res1(m, k);
  Matrix<return_type> res2(m, k);
  Matrix<return_type> chk(m, k);
  Matrix<scalar_type> gauge(m, k);

  randm(a);
  randm(b);

  // Test matrix-matrix prod
  res1   = prod4(a, b);

  // Test matrix-vector prod
  for (index_type i=0; i<k; ++i)
    res2.col(i) = prod4(a, b.col(i));

  chk   = ref::prod(a, b);
  gauge = ref::prod(mag(a), mag(b));

  for (index_type i=0; i<gauge.size(0); ++i)
    for (index_type j=0; j<gauge.size(1); ++j)
      if (!(gauge(i, j) > scalar_type()))
	gauge(i, j) = scalar_type(1);

#if VERBOSE
  cout << "a     =\n" << a;
  cout << "b     =\n" << b;
#endif

  check_prod( res1, chk, gauge );
  check_prod( res2, chk, gauge );
}



template <typename T0,
	  typename T1>
void
prod_cases()
{
  test_prod3_rand<T0, T1>();
  test_prod4_rand<T0, T1>();
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  Precision_traits<float>::compute_eps();
  Precision_traits<double>::compute_eps();


  prod_cases<float,  float>();

  prod_cases<complex<float>, complex<float> >();
  prod_cases<float,          complex<float> >();
  prod_cases<complex<float>, float          >();

#if VSIP_IMPL_TEST_DOUBLE
  prod_cases<double, double>();
  prod_cases<float,  double>();
  prod_cases<double, float>();
#endif
}

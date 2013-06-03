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
#include <test.hpp>
#include <test/ref/matvec.hpp>
#include "prod.hpp"

using namespace ovxx;

// This is a somewhat arbitrary measure, but generally works.  Some
// platforms using different rounding modes or non-IEEE-compliant math
// may require a more relaxed threshold (try -80 instead).
#define ERROR_DB_THRESHOLD  -100.0f


/***********************************************************************
  Test Definitions
***********************************************************************/

/// Test matrix-matrix, matrix-vector, and vector-matrix products.

template <typename T0,
	  typename T1,
	  typename OrderR,
	  typename Order0,
	  typename Order1>
void
test_prod_rand(length_type m, length_type n, length_type k)
{
  typedef typename Promotion<T0, T1>::type return_type;
  typedef typename scalar_of<return_type>::type scalar_type;

  typedef Dense<2, T0, Order0>          block0_type;
  typedef Dense<2, T1, Order1>          block1_type;
  typedef Dense<2, return_type, OrderR> blockR_type;

  Matrix<T0, block0_type>          a(m, n);
  Matrix<T1, block1_type>          b(n, k);
  Matrix<return_type, blockR_type> res1(m, k);
  Matrix<return_type, blockR_type> res2(m, k);
  Matrix<return_type, blockR_type> res3(m, k);
  Matrix<return_type, blockR_type> chk(m, k);

  test::randm(a);
  test::randm(b);

  // Test matrix-matrix prod
  res1   = prod(a, b);

  // Test matrix-vector prod
  for (index_type i=0; i<k; ++i)
    res2.col(i) = prod(a, b.col(i));

  // Test vector-matrix prod
  for (index_type i=0; i<m; ++i)
    res3.row(i) = prod(a.row(i), b);

  // Compute reference
  chk   = test::ref::prod(a, b);

#if VERBOSE
  cout << "[" << m << "x" << n << "]  [" << 
    n << "x" << k << "]  [" << m << "x" << k << "]  "  << endl;
  cout << "a     =\n" << a;
  cout << "b     =\n" << b;
#endif

  test_assert(test::diff(res1, chk) < ERROR_DB_THRESHOLD);
  test_assert(test::diff(res2, chk) < ERROR_DB_THRESHOLD);
  test_assert(test::diff(res3, chk) < ERROR_DB_THRESHOLD);
}



template <typename T0,
	  typename T1,
	  typename OrderR,
	  typename Order0,
	  typename Order1>
void
prod_types_with_order()
{
  test_prod_rand<T0, T1, OrderR, Order0, Order1>(5, 5, 5);
  test_prod_rand<T0, T1, OrderR, Order0, Order1>(5, 7, 9);
  test_prod_rand<T0, T1, OrderR, Order0, Order1>(9, 5, 7);
  test_prod_rand<T0, T1, OrderR, Order0, Order1>(9, 7, 5);
}


template <typename T0,
	  typename T1>
void
prod_cases_with_order()
{
  prod_types_with_order<T0, T1, row2_type, row2_type, row2_type>();
  prod_types_with_order<T0, T1, row2_type, col2_type, row2_type>();
  prod_types_with_order<T0, T1, row2_type, row2_type, col2_type>();
  prod_types_with_order<T0, T1, col2_type, col2_type, col2_type>();
  prod_types_with_order<T0, T1, col2_type, row2_type, col2_type>();
  prod_types_with_order<T0, T1, col2_type, col2_type, row2_type>();
}


/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test::precision<float>::init();
  test::precision<double>::init();

  prod_cases_with_order<float,  float>();

  prod_cases_with_order<complex<float>, complex<float> >();
  prod_cases_with_order<float,          complex<float> >();
  prod_cases_with_order<complex<float>, float          >();

#if VSIP_IMPL_TEST_DOUBLE
  prod_cases_with_order<double, double>();
  prod_cases_with_order<float,  double>();
  prod_cases_with_order<double, float>();
#endif


  // Test a large matrix-matrix product (order > 80) to trigger
  // ATLAS blocking code.  If order < NB, only the cleanup code
  // gets exercised.
  test_prod_rand<float, float, row2_type, row2_type, row2_type>(256, 256, 256);
}

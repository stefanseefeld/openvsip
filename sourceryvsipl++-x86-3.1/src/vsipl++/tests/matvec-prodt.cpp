/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/matvec-prodt.cpp
    @author  Jules Bergmann
    @date    2005-09-12
    @brief   VSIPL++ Library: Unit tests for matrix products.
*/

/***********************************************************************
  Included Files
***********************************************************************/

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
          typename T1,
          typename OrderR,
          typename Order0,
          typename Order1>
void
test_prodt_rand(length_type m, length_type n, length_type k)
{
  typedef typename Promotion<T0, T1>::type return_type;
  typedef typename vsip::impl::scalar_of<return_type>::type scalar_type;

  typedef Dense<2, T0, Order0>          block0_type;
  typedef Dense<2, T1, Order1>          block1_type;
  typedef Dense<2, return_type, OrderR> blockR_type;

  Matrix<T0, block0_type> a(m, n);
  Matrix<T1, block1_type> b(k, n);
  Matrix<return_type, blockR_type> res1(m, k);
  Matrix<return_type, blockR_type> chk(m, k);
  Matrix<scalar_type> gauge(m, k);

  randm(a);
  randm(b);

  // Test matrix-matrix prod for transpose
  res1 = prodt(a, b);

  chk   = ref::prod(a, trans(b));
  gauge = ref::prod(mag(a), mag(trans(b)));

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
	  typename T1,
	  typename OrderR,
	  typename Order0,
	  typename Order1>
void
prodt_types_with_order()
{
  test_prodt_rand<T0, T1, OrderR, Order0, Order1>(5, 5, 5);
  test_prodt_rand<T0, T1, OrderR, Order0, Order1>(5, 7, 9);
  test_prodt_rand<T0, T1, OrderR, Order0, Order1>(9, 5, 7);
  test_prodt_rand<T0, T1, OrderR, Order0, Order1>(9, 7, 5);
}


template <typename T0,
	  typename T1>
void
prodt_cases_with_order()
{
  prodt_types_with_order<T0, T1, row2_type, row2_type, row2_type>();
  prodt_types_with_order<T0, T1, row2_type, row2_type, col2_type>();
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


  prodt_cases_with_order<float,  float>();

  prodt_cases_with_order<complex<float>, complex<float> >();
  prodt_cases_with_order<float,          complex<float> >();
  prodt_cases_with_order<complex<float>, float          >();

#if VSIP_IMPL_TEST_DOUBLE
  prodt_cases_with_order<double, double>();
  prodt_cases_with_order<float,  double>();
  prodt_cases_with_order<double, float>();
#endif
}

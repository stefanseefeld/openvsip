/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/matvec-prod-special.cpp
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

/// Test matrix-matrix products using sub-views

template <typename T>
void
test_mm_prod_subview( const length_type m, 
                      const length_type n, 
                      const length_type k )
{
  typedef typename Matrix<T>::subview_type matrix_subview_type;
  typedef typename vsip::impl::scalar_of<T>::type scalar_type;

  // non-unit strides - dense rows, non-dense columns
  {
    Matrix<T> aa(m, k*2, T());
    Matrix<T> bb(k, n*3, T());
    matrix_subview_type a = aa(Domain<2>(
                                 Domain<1>(0, 1, m), Domain<1>(0, 2, k)));
    matrix_subview_type b = bb(Domain<2>(
                                 Domain<1>(0, 1, k), Domain<1>(0, 3, n)));
    Matrix<T> res(m, n);
    Matrix<T> chk(m, n);
    Matrix<scalar_type> gauge(m, n);

    randm(a);
    randm(b);

    res = prod( a, b );
    chk = ref::prod( a, b );
    gauge = ref::prod(mag(a), mag(b));

    for (index_type i=0; i<gauge.size(0); ++i)
      for (index_type j=0; j<gauge.size(1); ++j)
        if (!(gauge(i, j) > scalar_type()))
          gauge(i, j) = scalar_type(1);

    check_prod( res, chk, gauge );
  }

  // non-unit strides - non-dense rows, dense columns
  {
    Matrix<T> aa(m*2, k, T());
    Matrix<T> bb(k*3, n, T());
    matrix_subview_type a = aa(Domain<2>( 
                                 Domain<1>(0, 2, m), Domain<1>(0, 1, k)));
    matrix_subview_type b = bb(Domain<2>(
                                 Domain<1>(0, 3, k), Domain<1>(0, 1, n)));

    Matrix<T> res(m, n);
    Matrix<T> chk(m, n);
    Matrix<scalar_type> gauge(m, n);

    randm(a);
    randm(b);

    res = prod( a, b );
    chk = ref::prod( a, b );
    gauge = ref::prod(mag(a), mag(b));

    for (index_type i=0; i<gauge.size(0); ++i)
      for (index_type j=0; j<gauge.size(1); ++j)
        if (!(gauge(i, j) > scalar_type()))
          gauge(i, j) = scalar_type(1);

    check_prod( res, chk, gauge );
  }
}


/// Test matrix-matrix products using split-complex format

template <typename T>
void 
test_mm_prod_complex_split(  const length_type m, 
                             const length_type n, 
                             const length_type k )
{
  typedef vsip::impl::Strided<2, complex<T>,
    vsip::Layout<2, row2_type, vsip::dense, vsip::split_complex> > split_type;
  typedef typename vsip::impl::scalar_of<T>::type scalar_type;
  
  Matrix<complex<T>, split_type> a(m, k);
  Matrix<complex<T>, split_type> b(k, n);
  Matrix<complex<T>, split_type> res(m, n, T());

  randm(a);
  randm(b);

  // call prod()'s underlying interface directly
  res = prod(a, b);

  // compute a reference matrix using interleaved (default) layout
  Matrix<complex<T> > aa(m, k);
  Matrix<complex<T> > bb(k, n);
  aa = a;
  bb = b;

  Matrix<complex<T> > chk(m, n);
  Matrix<scalar_type> gauge(m, n);
  chk = ref::prod( aa, bb );
  gauge = ref::prod( mag(aa), mag(bb) );

  for (index_type i=0; i<gauge.size(0); ++i)
    for (index_type j=0; j<gauge.size(1); ++j)
      if (!(gauge(i, j) > scalar_type()))
        gauge(i, j) = scalar_type(1);

  check_prod( res, chk, gauge );
}



template <typename T>
void 
prod_special_cases()
{
  test_mm_prod_subview<T>(5, 7, 3);
  test_mm_prod_complex_split<T>(5, 7, 3);
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

  prod_special_cases<float>();

#if VSIP_IMPL_TEST_DOUBLE
  prod_special_cases<double>();
#endif
}

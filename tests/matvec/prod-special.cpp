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

/// Test matrix-matrix products using sub-views

template <typename T>
void
test_mm_prod_subview( const length_type m, 
                      const length_type n, 
                      const length_type k )
{
  typedef typename Matrix<T>::subview_type matrix_subview_type;
  typedef typename scalar_of<T>::type scalar_type;

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

    test::randm(a);
    test::randm(b);

    res = prod( a, b );
    chk = test::ref::prod( a, b );
    gauge = test::ref::prod(mag(a), mag(b));

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

    test::randm(a);
    test::randm(b);

    res = prod( a, b );
    chk = test::ref::prod( a, b );
    gauge = test::ref::prod(mag(a), mag(b));

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
  typedef Strided<2, complex<T>,
    vsip::Layout<2, row2_type, vsip::dense, vsip::split_complex> > split_type;
  typedef typename scalar_of<T>::type scalar_type;
  
  Matrix<complex<T>, split_type> a(m, k);
  Matrix<complex<T>, split_type> b(k, n);
  Matrix<complex<T>, split_type> res(m, n, T());

  test::randm(a);
  test::randm(b);

  // call prod()'s underlying interface directly
  res = prod(a, b);

  // compute a reference matrix using interleaved (default) layout
  Matrix<complex<T> > aa(m, k);
  Matrix<complex<T> > bb(k, n);
  aa = a;
  bb = b;

  Matrix<complex<T> > chk(m, n);
  Matrix<scalar_type> gauge(m, n);
  chk = test::ref::prod( aa, bb );
  gauge = test::ref::prod( mag(aa), mag(bb) );

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

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test::precision<float>::init();
  test::precision<double>::init();

  prod_special_cases<float>();

#if VSIP_IMPL_TEST_DOUBLE
  prod_special_cases<double>();
#endif
}

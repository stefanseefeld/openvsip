/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/freqswap.cpp
    @author  Don McCoy
    @date    2005-12-01
    @brief   VSIPL++ Library: Frequency swap unit tests [signal.freqswap]
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

using namespace vsip;
using vsip_csl::equal;

/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
void
test_vector_freqswap( length_type m )
{
  Vector<T> a(m);

  Rand<T> rgen(0);
  a = rgen.randu(m);

  Vector<T> b(m);
  Vector<T> c(m);
  b = vsip::freqswap(a);
  c = a; c = vsip::freqswap(c);

  for ( index_type i = 0; i < m; i++ )
  {
    test_assert(equal( b.get(i), a.get(((m+1)/2 + i) % m ) ));
    test_assert(equal( c.get(i), a.get(((m+1)/2 + i) % m ) ));
  }
}



template <typename T>
void
test_real_subview_vector_freqswap( length_type m )
{
  Vector<complex<T> > a(m);

  a.real() = ramp<T>(0, 1, m);
  a.imag() = T(0);

  Vector<T> b(m);
  Vector<complex<T> > c(m, T());
  b = vsip::freqswap(a.real());
  c.real() = a.real(); c.real() = vsip::freqswap(c.real());

  for ( index_type i = 0; i < m; i++ )
  {
    test_assert(equal( b.get(i), a.real().get(((m+1)/2 + i) % m ) ));
    test_assert(equal( c.real().get(i), a.real().get(((m+1)/2 + i) % m ) ));
  }
}



template <typename T1,
	  typename T2>
void
test_diff_type_vector_freqswap( length_type m )
{
  Vector<T1> a(m);

  a = ramp<T1>(0, 1, m);

  Vector<T2> b(m);
  b = vsip::freqswap(a);

  for ( index_type i = 0; i < m; i++ )
  {
    test_assert(equal( b.get(i), (T2)a.get(((m+1)/2 + i) % m ) ));
  }
}


template <typename T>
void
test_matrix_freqswap( length_type m, length_type n )
{
  Matrix<T> a(m, n);

  Rand<T> rgen(0);
  a = rgen.randu(m, n);

  Matrix<T> b(m, n);
  Matrix<T> c(m, n);
  b = vsip::freqswap(a);
  c = a; c = vsip::freqswap(c);

  for ( index_type i = 0; i < m; i++ )
    for ( index_type j = 0; j < n; j++ )
    {
      test_assert(equal( b.get(i, j),
               a.get(((m+1)/2 + i) % m, ((n+1)/2 + j) % n ) ));
      test_assert(equal( c.get(i, j),
               a.get(((m+1)/2 + i) % m, ((n+1)/2 + j) % n ) ));
    }
}



template <typename T1,
	  typename T2>
void
test_diff_type_matrix_freqswap( length_type m, length_type n )
{
  Matrix<T1> a(m, n);

  Rand<T1> rgen(0);
  a = rgen.randu(m, n);

  Matrix<T2> b(m, n);
  b = vsip::freqswap(a);

  for ( index_type i = 0; i < m; i++ )
    for ( index_type j = 0; j < n; j++ )
    {
      test_assert(equal( b.get(i, j),
			 (T2)a.get(((m+1)/2 + i) % m, ((n+1)/2 + j) % n ) ));
    }
}



template <typename T>
void
cases_by_type()
{
  test_vector_freqswap<T>( 8 );
  test_vector_freqswap<T>( 9 );
  test_vector_freqswap<T>( 33 );

  test_real_subview_vector_freqswap<T>( 8 );
  test_real_subview_vector_freqswap<T>( 9 );
  test_real_subview_vector_freqswap<T>( 33 );

  test_matrix_freqswap<T>( 4, 4 );
  test_matrix_freqswap<T>( 4, 5 );
  test_matrix_freqswap<T>( 5, 4 );
  test_matrix_freqswap<T>( 5, 5 );
}
  



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_diff_type_vector_freqswap<float, double>(8);
  test_diff_type_matrix_freqswap<float, double>(4, 4);

  cases_by_type<float>();
#if VSIP_IMPL_TEST_DOUBLE
  cases_by_type<double>();
#endif // VSIP_IMPL_TEST_DOUBLE
#if VSIP_IMPL_TEST_LONG_DOUBLE
  cases_by_type<long double>();
#endif // VSIP_IMPL_TEST_LONG_DOUBLE

  return EXIT_SUCCESS;
}

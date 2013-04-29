//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <cassert>
#include <math.h>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>

#include <vsip_csl/test.hpp>
#include "test-random.hpp"
#include <vsip_csl/output.hpp>
#include <vsip_csl/ref_matvec.hpp>

using namespace std;
using namespace vsip;
namespace ref = vsip_csl::ref;
using vsip_csl::equal;



/***********************************************************************
  Macros
***********************************************************************/

// 080314: For MCOE csr1610, these macros are not defined by GCC
//         math.h/cmath (but are defined by GHS math.h/cmath).

#if _MC_EXEC && __GNUC__
#  define M_E        2.718281828459045235360
#  define M_LN2      0.69314718055994530942
#  define M_SQRT2    1.41421356237309504880
#  define M_LN10     2.30258509299404568402
#  define M_LOG2E    1.442695040888963407
#endif



/***********************************************************************
  Definitions
***********************************************************************/


// Error metric between two Vectors

template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
double
error_db(
  const_Vector<T1, Block1> v1,
  const_Vector<T2, Block2> v2)
{
  double refmax = 0.0;
  double maxsum = -250;
  double sum;

  Index<1> idx;

  refmax = maxval(magsq(v1), idx);

  for (index_type i=0; i<v1.size(); ++i)
  {
    double val = magsq(v1.get(i) - v2.get(i));

    if (val < 1.e-20)
      sum = -201.;
    else
      sum = 10.0 * log10(val/(2.0*refmax));

    if (sum > maxsum)
      maxsum = sum;
  }

  return maxsum;
}


// Error metric between two Matrices

template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
double
error_db(
  const_Matrix<T1, Block1> v1,
  const_Matrix<T2, Block2> v2)
{
  double maxsum = -250;
  for (unsigned i = 0; i < v1.size(0); ++i)
  {
    double sum = error_db(v1.row(i), v2.row(i));
    if (sum > maxsum)
      maxsum = sum;
  }
  return maxsum;
}


template <typename T>
void
Check_gem_results( Matrix<T> actual, Matrix<T> expected )
{
  test_assert( error_db(actual, expected) < -100 );
}


template <typename T>
void
Test_gemp( T alpha, T beta, length_type M, length_type P, length_type N )
{
  Matrix<T> a (M, P);
  Matrix<T> b (P, N);
  Matrix<T> c (M, N);
  Matrix<T> d (M, N);

  Matrix<T> a_t (P, M);
  Matrix<T> b_t (N, P);

  // fill in unique values for each element of a, b and c
  randm(a);
  randm(b);
  randm(c);
  a_t = a.transpose();
  b_t = b.transpose();


  // compute the expected result for d
  index_type row;
  index_type col;
  for ( row = 0; row < M; ++row )
    for ( col = 0; col < N; ++col )
    {
      T dot = 0;
      for ( index_type i = 0; i < P; ++i )
        dot += a.get(row, i) * b.get(i, col);
      d.put( row, col, alpha * dot + beta * c(row, col) );
    }

  // compute the actual result (updated in c)
  gemp<mat_ntrans, mat_ntrans>(alpha, a, b, beta, c);

  Check_gem_results( c, d );


  // re-compute the result with remaining types
  
  // trans, no trans
  // compute the expected result for d
  for ( row = 0; row < M; ++row )
    for ( col = 0; col < N; ++col )
    {
      T dot = 0;
      for ( index_type i = 0; i < P; ++i )
        dot += a_t.get(i, row) * b.get(i, col);
      d.put( row, col, alpha * dot + beta * c(row, col) );
    }

  // compute the actual result (updated in c)
  gemp<mat_trans, mat_ntrans>(alpha, a_t, b, beta, c);

  Check_gem_results( c, d );

  
  // no trans, trans
  // compute the expected result for d
  for ( row = 0; row < M; ++row )
    for ( col = 0; col < N; ++col )
    {
      T dot = 0;
      for ( index_type i = 0; i < P; ++i )
        dot += a.get(row, i) * b_t.get(col, i);
      d.put( row, col, alpha * dot + beta * c(row, col) );
    }

  // compute the actual result (updated in c)
  gemp<mat_ntrans, mat_trans>(alpha, a, b_t, beta, c);

  Check_gem_results( c, d );


  // trans, trans
  // compute the expected result for d
  for ( row = 0; row < M; ++row )
    for ( col = 0; col < N; ++col )
    {
      T dot = 0;
      for ( index_type i = 0; i < P; ++i )
        dot += a_t.get(i, row) * b_t.get(col, i);
      d.put( row, col, alpha * dot + beta * c(row, col) );
    }

  // compute the actual result (updated in c)
  gemp<mat_trans, mat_trans>(alpha, a_t, b_t, beta, c);

  Check_gem_results( c, d );


  // herm, trans
  // compute the expected result for d
  for ( row = 0; row < M; ++row )
    for ( col = 0; col < N; ++col )
    {
      T dot = 0;
      for ( index_type i = 0; i < P; ++i )
        dot += vsip_csl::impl_conj(a_t.get(i, row)) * b_t.get(col, i);
      d.put( row, col, alpha * dot + beta * c(row, col) );
    }

  // compute the actual result (updated in c)
  gemp<mat_herm, mat_trans>(alpha, a_t, b_t, beta, c);

  Check_gem_results( c, d );


  // ntrans, conj
  // compute the expected result for d
  for ( row = 0; row < M; ++row )
    for ( col = 0; col < N; ++col )
    {
      T dot = 0;
      for ( index_type i = 0; i < P; ++i )
        dot += a.get(row, i) * vsip_csl::impl_conj(b.get(i, col));
      d.put( row, col, alpha * dot + beta * c(row, col) );
    }

  // compute the actual result (updated in c)
  gemp<mat_ntrans, mat_conj>(alpha, a, b, beta, c);

  Check_gem_results( c, d );
}


template <typename T>
void
Test_gems( T alpha, T beta, length_type M, length_type /*P*/, length_type N )
{
  Matrix<T> a (M, N);
  Matrix<T> b (M, N);
  Matrix<T> c (M, N);
  Matrix<T> d (M, N);

  Matrix<T> a_t (N, M);


  // fill in unique values for each element of a and c
  randm(a);
  randm(b); // save copy for later
  c = b;


  // without trans
  // compute the expected result for d
  for ( index_type row = 0; row < M; ++row )
    for ( index_type col = 0; col < N; ++col )
      d.put( row, col, alpha * a.get(row, col) + beta * c(row, col) );

  // compute the actual result (updated in c)
  gems<mat_ntrans>(alpha, a, beta, c);

  Check_gem_results( c, d );


  // create the transposes of a and restore c
  c = b;
  a_t = a.transpose();

  // with trans
  // expected result for d will stay the same because now we use
  // the transpose of a and request that it take the transpose 
  // of that when computing the sum

  // compute the actual result (updated in c)
  gems<mat_trans>(alpha, a_t, beta, c);

  Check_gem_results( c, d );


  // restore c
  c = b;

  // with herm
  // compute the expected result for d
  for ( index_type row = 0; row < M; ++row )
    for ( index_type col = 0; col < N; ++col )
      d.put( row, col, alpha * vsip_csl::impl_conj(a_t.get(col, row)) + beta * c(row, col) );

  // compute the actual result (updated in c)
  gems<mat_herm>(alpha, a_t, beta, c);

  Check_gem_results( c, d );


  // restore c
  c = b;

  // with conj
  // compute the expected result for d
  for ( index_type row = 0; row < M; ++row )
    for ( index_type col = 0; col < N; ++col )
      d.put( row, col, alpha * vsip_csl::impl_conj(a.get(row, col)) + beta * c(row, col) );

  // compute the actual result (updated in c)
  gems<mat_conj>(alpha, a, beta, c);

  Check_gem_results( c, d );
}


template <typename T>
void
Test_gem_types( T alpha, T beta )
{
  // last 3 params are M, N, P (for M x N and N x P matricies)

  // generalized matrix product
  Test_gemp<T>( alpha, beta, 7, 3, 5 );
  Test_gemp<T>( alpha, beta, 7, 9, 5 );
  Test_gemp<T>( alpha, beta, 5, 9, 7 );
  Test_gemp<T>( alpha, beta, 5, 3, 7 );

  // generalized matrix sum
  Test_gems<T>( alpha, beta, 7, 3, 5 );
  Test_gems<T>( alpha, beta, 7, 9, 5 );
  Test_gems<T>( alpha, beta, 5, 9, 7 );
  Test_gems<T>( alpha, beta, 5, 3, 7 );
}



void
Test_cumsum()
{
  // simple sum of a vector containing scalars
  length_type const len = 5;
  Vector<scalar_f> v1( len );
  Vector<scalar_f> v2( len );
  scalar_f sum = 0;

  for ( index_type i = 0; i < len; ++i )
  {
    v1.put( i, i + 1 );
    sum += i + 1;
  }

  cumsum<0>( v1, v2 );
  test_assert( equal( sum, v2.get(len - 1) ) );


  // simple sum of a vector containing complex<scalars>
  Vector<cscalar_f> cv1( len );
  Vector<cscalar_f> cv2( len );
  cscalar_f csum = cscalar_f();

  for ( index_type i = 0; i < len; ++i )
  {
    cv1.put( i, complex<float>( i + 1, i + 1 ) );
    csum += complex<float>( i + 1, i + 1 );
  }

  cumsum<0>( cv1, cv2 );
  test_assert( equal( csum, cv2.get(len - 1) ) );


  // sum of a matrix using scalars
  length_type const rows = 5;
  length_type const cols = 7;
  Matrix<scalar_f> m1( rows, cols );
  Matrix<scalar_f> m2( rows, cols );
  scalar_f colsum[7];
  scalar_f rowsum[5];

  for ( index_type i = 0; i < rows; ++i )
  {
    rowsum[i] = 0;
    for ( index_type j = 0; j < cols; ++j )
    {
      m1.put( i, j, i + 1 + j * rows );
      rowsum[i] += i + 1 + j * rows;
    }
  }

  for ( index_type j = 0; j < cols; ++j )
  {
    colsum[j] = 0;
    for ( index_type i = 0; i < rows; ++i )
      colsum[j] += i + 1 + j * rows;
  }

  // sum across rows of a matrix
  cumsum<0>( m1, m2 );
  for ( index_type i = 0; i < rows; ++i )
    test_assert( equal( rowsum[i], m2.get(i, cols - 1) ) );


  // sum across columns of a matrix
  cumsum<1>( m1, m2 );
  for ( index_type j = 0; j < cols; ++j )
    test_assert( equal( colsum[j], m2.get(rows - 1, j) ) );
}  


template <typename T0,
          typename T3>
void
Test_modulate( const length_type m )
{
  index_type rows = 2;
  Matrix<T0> v(rows, m);
  Matrix<complex<T3> > w(rows, m, complex<float>());
  Matrix<complex<T3> > r(rows, m);

  T3 nu = VSIP_IMPL_PI / 2;
  T3 phi = 0.123;
  T3 phase = phi;

  randm(v);

  for ( index_type i = 0; i < rows; ++i )
  {
    phase = vsip::modulate(v.row(i), nu, phase, w.row(i));

    for ( index_type j = 0; j < m; ++j )
      r.put( i, j, v.get(i, j) * exp(complex<T3>(0, (i * m + j) * nu + phi)) );
  }

  test_assert( error_db(r, w) < -100 );
}


template <typename T>
void
Test_outer( T alpha, const length_type m, const length_type n )
{
  {
    Vector<T> a(m, T());
    Vector<T> b(n, T());
    Matrix<T> r(m, n, T());
    Matrix<T> c1(m, n, T(2));
    Matrix<T, Dense<2, T, col2_type> > c2(m, n, T(2));

    randv(a);
    randv(b);

    c1 = outer(alpha, a, b);
    vsip::impl::outer(alpha, a, b, c2);
    r = ref::outer(alpha * a, b);

    for ( vsip::index_type i = 0; i < r.size(0); ++i )
      for ( vsip::index_type j = 0; j < r.size(1); ++j )
      {
        test_assert( equal( r.get(i, j), c1.get(i, j) ) );
        test_assert( equal( r.get(i, j), c2.get(i, j) ) );
      }
  }
}


template <typename T>
void
modulate_cases( const length_type m )
{
  Test_modulate<T,          T>( m );
  Test_modulate<complex<T>, T>( m );
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // Test Vector Kronecker
  {
    Vector<> v1(4, 7.);
    Vector<> v2(5, 11.);
    Matrix<> result = kron(2., v1, v2);
    test_assert(result.size(0) == 5);
    test_assert(result.size(1) == 4);

    for (index_type i = 0; i != result.size(0); ++i)
      for (index_type j = 0; j != result.size(1); ++j)
        test_assert(equal(result.get(i, j), 2.f * v1.get(j)*v2.get(i)));
  }
  // Test Matrix-Matrix Kronecker

  Matrix<scalar_f>
    matrix_m( 2, 3, static_cast<scalar_f>(7.0) );
  Matrix<scalar_f>
    matrix_n( 4, 5, static_cast<scalar_f>(11.0) );
  Matrix<>
    kron_mn(kron (static_cast<scalar_f>(2.0), matrix_m, matrix_n));

  test_assert( kron_mn.size(0) == 2 * 4 );
  test_assert( kron_mn.size(1) == 3 * 5 );

  for ( index_type a = 2 * 4; a-- > 0; )
    for ( index_type b = 3 * 5; b-- > 0; )
      test_assert( equal( kron_mn.get( a, b ),
                static_cast<scalar_f>(7 * 11 * 2.0) ) );


  // Test generalized matrix operations

  // params: alpha, beta
  Test_gem_types<float>( M_E, VSIP_IMPL_PI );

  Test_gem_types<complex<float> >
    ( complex<float>(M_LN2, -M_SQRT2), complex<float>(M_LOG2E, M_LN10) );

#if VSIP_IMPL_TEST_DOUBLE
  Test_gem_types<double>( -M_E, -VSIP_IMPL_PI );

  Test_gem_types<complex<double> >
    ( complex<float>(M_LN2, -M_SQRT2), complex<float>(M_LOG2E, M_LN10) );
#endif


  // misc functions
  
  Test_cumsum();

  modulate_cases<float>(10);
#if VSIP_IMPL_TEST_DOUBLE
  modulate_cases<double>(32);
#endif
#if VSIP_IMPL_HAVE_COMPLEX_LONG_DOUBLE && VSIP_IMPL_TEST_LONG_DOUBLE
  modulate_cases<long double>(16);
#endif

  Test_outer<float>( static_cast<float>(VSIP_IMPL_PI), 3, 3 );
  Test_outer<float>( static_cast<float>(VSIP_IMPL_PI), 5, 7 );
  Test_outer<float>( static_cast<float>(VSIP_IMPL_PI), 7, 5 );
  Test_outer<complex<float> >( complex<float>(VSIP_IMPL_PI, M_E), 3, 3 );
  Test_outer<complex<float> >( complex<float>(VSIP_IMPL_PI, M_E), 5, 7 );
  Test_outer<complex<float> >( complex<float>(VSIP_IMPL_PI, M_E), 7, 5 );
#if VSIP_IMPL_TEST_DOUBLE
  Test_outer<double>( static_cast<double>(VSIP_IMPL_PI), 3, 3 );
  Test_outer<double>( static_cast<double>(VSIP_IMPL_PI), 5, 7 );
  Test_outer<double>( static_cast<double>(VSIP_IMPL_PI), 7, 5 );
  Test_outer<complex<double> >( complex<double>(VSIP_IMPL_PI, M_E), 3, 3 );
  Test_outer<complex<double> >( complex<double>(VSIP_IMPL_PI, M_E), 5, 7 );
  Test_outer<complex<double> >( complex<double>(VSIP_IMPL_PI, M_E), 7, 5 );
#endif

  return 0;
}


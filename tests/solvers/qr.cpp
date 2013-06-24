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
#include <vsip/solvers.hpp>
#include <vsip/parallel.hpp>
#include <test.hpp>
#include <storage.hpp>
#include "common.hpp"

#define VERBOSE        0
#define DO_SWEEP       0
#define NORMAL_EPSILON 0

#ifdef VSIP_IMPL_HAVE_LAPACK
#  define EXPECT_FULL_COVERAGE 1
#else
#  define EXPECT_FULL_COVERAGE 0
#endif

using namespace ovxx;

// Simple covsol test

template <typename T>
void
test_covsol_diag(
  length_type  m,
  length_type  n,
  length_type  p,
  storage_type st
  )
{
  test_assert(m >= n);

  Matrix<T> a(m, n);
  Matrix<T> b(n, p);
  Matrix<T> x(n, p);

  a        = T();
  a.diag() = T(1);
  if (n > 0) a(0, 0)  = Test_traits<T>::value1();
  if (n > 2) a(2, 2)  = Test_traits<T>::value2();
  if (n > 3) a(3, 3)  = Test_traits<T>::value3();

  qrd<T, by_reference> qr(m, n, st);

  test_assert(qr.rows()     == m);
  test_assert(qr.columns()  == n);
  test_assert(qr.qstorage() == st);

  qr.decompose(a);

  for (index_type i=0; i<p; ++i)
    test::ramp(b.col(i), T(1), T(i));
  if (p > 1)
    b.col(1) += Test_traits<T>::offset();

  qr.covsol(b, x);

#if VERBOSE
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
#endif

  for (index_type c=0; c<p; ++c)
    for (index_type r=0; r<n; ++r)
      test_assert(equal(b(r, c), tconj<T>(a(r, r)) * a(r, r) * x(r, c)));
}



template <typename T,
	  typename MapT>
void
test_covsol_random(
  length_type  m,
  length_type  n,
  length_type  p,
  storage_type st)
{
  test_assert(m >= n);

  typedef Dense<2, T, row2_type, MapT> block_type;

  MapT map = create_map<2, MapT>();

  Matrix<T, block_type> a(m, n, map);
  Matrix<T, block_type> b(n, p, map);
  Matrix<T, block_type> x(n, p, map);

  test::randm(a);

  qrd<T, by_reference> qr(m, n, st);

  test_assert(qr.rows()     == m);
  test_assert(qr.columns()  == n);
  test_assert(qr.qstorage() == st);

  qr.decompose(a);

  test::randm(b);

  qr.covsol(b, x);


  Matrix<T> c(n, n);
  Matrix<T> chk(n, p);

  prodh(a, a, c);
  prod(c, x, chk);

  float err = prod_check(c, x, b);

#if VERBOSE
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
  cout << "chk = " << endl << chk << endl;
  cout << "covsol<" << Type_name<T>::name()
       << ">(" << m << ", " << n << ", " << p << "): " << err << endl;
#endif

  if (err > 10.0)
  {
    for (index_type r=0; r<n; ++r)
      for (index_type c=0; c<p; ++c)
	test_assert(equal(b(r, c), chk(r, c)));
  }
}



template <typename T>
void covsol_cases(storage_type st, true_type)
{
  test_covsol_diag<T>(1,   1, 2, st);
  test_covsol_diag<T>(5,   5, 2, st);
  test_covsol_diag<T>(6,   6, 2, st);
  test_covsol_diag<T>(17, 17, 2, st);

  test_covsol_diag<T>(1,   1, 3, st);
  test_covsol_diag<T>(5,   5, 3, st);
  test_covsol_diag<T>(17, 17, 3, st);

  test_covsol_diag<T>(3,   1, 3, st);
  test_covsol_diag<T>(5,   3, 3, st);
  test_covsol_diag<T>(17, 11, 3, st);

  test_covsol_random<T, Local_map>(1,   1, 2, st);
  test_covsol_random<T, Local_map>(5,   5, 2, st);
  test_covsol_random<T, Local_map>(17, 17, 2, st);

  test_covsol_random<T, Local_map>(1,   1, 3, st);
  test_covsol_random<T, Local_map>(5,   5, 3, st);
  test_covsol_random<T, Local_map>(17, 17, 3, st);

  test_covsol_random<T, Local_map>(3,   1, 3, st);
  test_covsol_random<T, Local_map>(5,   3, 3, st);
  test_covsol_random<T, Local_map>(17, 11, 3, st);

#if DO_SWEEP
  for (index_type i=1; i<100; i+= 8)
    for (index_type j=1; j<10; j += 4)
    {
      test_covsol_random<T, Local_map>(i,   i,   j+1, st);
      test_covsol_random<T, Local_map>(i+1, i+1, j,   st);
      test_covsol_random<T, Local_map>(i+2, i+2, j+2, st);
    }
#endif
}



template <typename T>
void covsol_cases(storage_type, false_type)
{
  test_assert(!EXPECT_FULL_COVERAGE);
}



// Front-end function for covsol_cases.

// This function dispatches to either real set of tests or an empty
// function depending on whether the QR backends configured in support
// value type T.  (Not all QR backends support all value types).

template <typename T>
void covsol_cases(storage_type st)
{
  using namespace ovxx::dispatcher;

  covsol_cases<T>(st, 
    integral_constant<bool, is_operation_supported<op::qrd, T>::value>());
}



/***********************************************************************
  Linear Least Squares tests
***********************************************************************/

template <typename T>
void
test_lsqsol_diag(
  length_type  m,
  length_type  n,
  length_type  p,
  storage_type st)
{
  test_assert(m >= n);

  Matrix<T> a(m, n);
  Matrix<T> x(n, p);
  Matrix<T> b(m, p);

  a        = T();
  a.diag() = T(1);
  if (n > 0) a(0, 0)  = Test_traits<T>::value1();
  if (n > 2) a(2, 2)  = Test_traits<T>::value2();
  if (n > 3) a(3, 3)  = Test_traits<T>::value3();

  qrd<T, by_reference> qr(m, n, st);

  test_assert(qr.rows()     == m);
  test_assert(qr.columns()  == n);
  test_assert(qr.qstorage() == st);

  qr.decompose(a);

  for (index_type i=0; i<p; ++i)
    test::ramp(b.col(i), T(1), T(i));
  if (p > 1)
    b.col(1) += Test_traits<T>::offset();

  qr.lsqsol(b, x);

#if VERBOSE
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
#endif

  for (index_type c=0; c<p; ++c)
    for (index_type r=0; r<n; ++r)
      test_assert(equal(b(r, c),
		   a(r, r) * x(r, c)));
}



template <typename T,
	  typename MapT>
void
test_lsqsol_random(length_type  m,
		   length_type  n,
		   length_type  p,
		   storage_type st)
{
  test_assert(m >= n);

  typedef Dense<2, T, row2_type, MapT> block_type;

  MapT map = create_map<2, MapT>();

  Matrix<T, block_type> a(m, n, map);
  Matrix<T, block_type> x(n, p, map);
  Matrix<T, block_type> b(m, p, map);
  Matrix<T, block_type> chk(m, p, map);

  test::randm(a);
  test::randm(b);

  // If m > n, min || AX - B || may not be zero,
  // Need way to check that X is best solution
  //
  // In the meantime, limit rank of A, B to produce zero minimum.

  for (index_type i=n; i<m; ++i)
  {
    a.row(i) = T(i-n+2) * a.row(i-n);
    b.row(i) = T(i-n+2) * b.row(i-n);
  }

  qrd<T, by_reference> qr(m, n, st);

  test_assert(qr.rows()     == m);
  test_assert(qr.columns()  == n);
  test_assert(qr.qstorage() == st);

  qr.decompose(a);


  qr.lsqsol(b, x);

  prod(a, x, chk);
  float err = prod_check(a, x, b);

#if VERBOSE
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
  cout << "chk = " << endl << chk << endl;
  cout << "adiff = " << endl << mag(b-chk) << endl;
  cout << "rdiff1 = " << endl << mag((b-chk)/b) << endl;
  cout << "rdiff2 = " << endl << mag(b-chk)/mag(b) << endl;
  cout << "lsqsol<" << Type_name<T>::name()
       << ">(" << m << ", " << n << ", " << p << "): " << err << endl;
#endif

  typedef typename scalar_of<T>::type scalar_type;
#if NORMAL_EPSILON
  // These are almost_equal()'s normal epsilon.  They work fine for Lapack.
  scalar_type rel_epsilon = 1e-3;
  scalar_type abs_epsilon = 1e-5;
  float       err_bound   = 10.0;
#else
  // These are looser bounds.  They are necessary for SAL.
  scalar_type rel_epsilon = 1e-3;
  scalar_type abs_epsilon = 1e-5;
  float       err_bound   = 50.0;
#endif

  if (err > err_bound)
  {
    for (index_type r=0; r<m; ++r)
      for (index_type c=0; c<p; ++c)
	test_assert(almost_equal(b.get(r, c), chk.get(r, c),
				 rel_epsilon, abs_epsilon));
  }

  // test_assert(err < 10.0);
}



template <typename T>
void lsqsol_cases(storage_type st, true_type)
{
  test_lsqsol_diag<T>(1,   1, 2, st);
  test_lsqsol_diag<T>(5,   5, 2, st);
  test_lsqsol_diag<T>(6,   6, 2, st);
  test_lsqsol_diag<T>(17, 17, 2, st);

  test_lsqsol_diag<T>(1,   1, 3, st);
  test_lsqsol_diag<T>(5,   5, 3, st);
  test_lsqsol_diag<T>(17, 17, 3, st);

  test_lsqsol_diag<T>(3,   1, 3, st);
  test_lsqsol_diag<T>(5,   3, 3, st);
  test_lsqsol_diag<T>(17, 11, 3, st);

  test_lsqsol_random<T, Local_map>(1,   1, 2, st);
  test_lsqsol_random<T, Local_map>(5,   5, 2, st);
  test_lsqsol_random<T, Local_map>(17, 17, 2, st);

  test_lsqsol_random<T, Local_map>(1,   1, 3, st);
  test_lsqsol_random<T, Local_map>(5,   5, 3, st);
  test_lsqsol_random<T, Local_map>(17, 17, 3, st);

  test_lsqsol_random<T, Local_map>(3,   1, 3, st);
  test_lsqsol_random<T, Local_map>(5,   3, 3, st);
  test_lsqsol_random<T, Local_map>(17, 11, 3, st);

#if DO_SWEEP
  for (index_type i=1; i<100; i+= 8)
    for (index_type j=1; j<10; j += 4)
    {
      test_lsqsol_random<T, Local_map>(i,   i,   j+1, st);
      test_lsqsol_random<T, Local_map>(i+1, i+1, j,   st);
      test_lsqsol_random<T, Local_map>(i+2, i+2, j+2, st);
    }
#endif
}



template <typename T>
void lsqsol_cases(storage_type, false_type)
{
  test_assert(!EXPECT_FULL_COVERAGE);
}



// Front-end function for lsqsol_cases.

template <typename T>
void lsqsol_cases(storage_type st)
{
  using namespace ovxx::dispatcher;

  lsqsol_cases<T>(st,
    integral_constant<bool, is_operation_supported<op::qrd, T>::value>());
}



/***********************************************************************
  Rsol tests
***********************************************************************/

template <typename T,
	  typename MapT>
void
test_rsol_diag(length_type  m,
	       length_type  n,
	       length_type  p,
	       storage_type st)
{
  test_assert(m >= n);

  typedef Dense<2, T, row2_type, MapT> block_type;
  
  MapT map = create_map<2, MapT>();

  Matrix<T, block_type> a(m, n, map);
  Matrix<T, block_type> x(n, p, map);
  Matrix<T, block_type> b(n, p, map);

  a        = T();
  // a.diag() = T(1);
  for (index_type i=0; i<min(m, n); ++i)
    a.put(i, i, T(1));
  if (n > 0) a(0, 0)  = Test_traits<T>::value1();
  if (n > 2) a(2, 2)  = Test_traits<T>::value2();
  if (n > 3) a(3, 3)  = Test_traits<T>::value3();

  qrd<T, by_reference> qr(m, n, st);

  test_assert(qr.rows()     == m);
  test_assert(qr.columns()  == n);
  test_assert(qr.qstorage() == st);

  qr.decompose(a);

  // -------------------------------------------------------------------
  // Check prodq()
  //   For real T, Q should be identity.
  //   For complex<T>, Q should be unitary
  // (For complex, we can use Q to check rsol)

  // Currently, we can only check rsol if doing a full QR, or if
  // doing a thing QR when m == n.
  if (st == qrd_saveq || (st == qrd_saveq1 && m == n))
  {
    Matrix<T, block_type> I(m, m, T(), map);
    // I.diag() = T(1);
    for (index_type i=0; i<m; ++i)
      I.put(i, i, T(1));
    Matrix<T, block_type> qi(m, m, map);
    Matrix<T, block_type> iq(m, m, map);
    Matrix<T, block_type> qtq(m, m, map);

    // First, check multiply w/identity from left-side:
    //   Q I = qi
    qr.template prodq<mat_ntrans, mat_lside>(I, qi);

    for (index_type i=0; i<m; ++i)
      for (index_type j=0; j<m; ++j)
	if (i == j)
	  test_assert(equal(qi(i, j) * tconj<T>(qi(i, j)), T(1)));
	else
	  test_assert(equal(qi(i, j), T()));

    // Next, check multiply w/identity from right-side:
    //   I Q = iq
    // (should get same answer as qi)
    qr.template prodq<mat_ntrans, mat_rside>(I, iq);

    for (index_type i=0; i<m; ++i)
      for (index_type j=0; j<m; ++j)
      {
	if (i == j)
	  test_assert(equal(iq(i, j) * tconj<T>(qi(i, j)), T(1)));
	else
	  test_assert(equal(iq(i, j), T()));
	test_assert(equal(iq(i, j), qi(i, j)));
      }

    // Next, check hermitian multiply w/Q from left-side:
    //   Q' (qi) = I
    //   Q' Q    = I
    mat_op_type const tr = is_complex<T>::value ? mat_herm : mat_trans;
    qr.template prodq<tr, mat_lside>(qi, qtq);

    // Result should be I
    for (index_type i=0; i<m; ++i)
      for (index_type j=0; j<m; ++j)
	test_assert(equal(qtq(i, j), I(i, j)));

  
    // -----------------------------------------------------------------
    // Check rsol()

    for (index_type i=0; i<p; ++i)
      test::ramp(b.col(i), T(1), T(i));
    if (p > 1) b.col(1) += Test_traits<T>::offset();
    
    T alpha = T(2);

    qr.template rsol<mat_ntrans>(b, alpha, x);

#if VERBOSE
    cout << "a = " << endl << a << endl;
    cout << "x = " << endl << x << endl;
    cout << "b = " << endl << b << endl;
    cout << "qi = " << endl << qi << endl;
#endif

    // a * x = alpha * Q * b

    for (index_type i=0; i<b.size(0); ++i)
      for (index_type j=0; j<b.size(1); ++j)
	test_assert(equal(alpha * qi(i, i) * b(i, j),
			  a(i, i) * x(i, j)));
  }
}


template <typename T>
void
rsol_cases(storage_type st, true_type)
{
  test_rsol_diag<T, Local_map>( 1,   1, 2, st);
  test_rsol_diag<T, Local_map>( 5,   4, 2, st);
  test_rsol_diag<T, Local_map>( 5,   5, 2, st);
  test_rsol_diag<T, Local_map>( 6,   6, 2, st);
  test_rsol_diag<T, Local_map>(17,  17, 2, st);
  test_rsol_diag<T, Local_map>(17,  11, 2, st);

  test_rsol_diag<T, Local_map>( 5,   2, 2, st);
  test_rsol_diag<T, Local_map>( 5,   3, 2, st);
  test_rsol_diag<T, Local_map>( 5,   4, 2, st);
  test_rsol_diag<T, Local_map>( 11,  5, 2, st);
}



template <typename T>
void
rsol_cases(storage_type, false_type)
{
  test_assert(!EXPECT_FULL_COVERAGE);
}



// Front-end function for rsol_cases.

template <typename T>
void rsol_cases(storage_type st)
{
  using namespace ovxx::dispatcher;

  rsol_cases<T>(st,
    integral_constant<bool, is_operation_supported<op::qrd, T>::value>());
}



/***********************************************************************
  covsol function using rsol tests
***********************************************************************/

template <return_mechanism_type ReturnMechanism>
struct Covsol_class;

template <>
struct Covsol_class<by_reference>
{

  template <typename T,
	    typename Block0,
	    typename Block1,
	    typename Block2>
  static Matrix<T, Block2>
  covsol(Matrix<T, Block0>       a,
	 const_Matrix<T, Block1> b,
	 Matrix<T, Block2>       x)
  {
    length_type m = a.size(0);
    length_type n = a.size(1);
    length_type p = b.size(1);
    
    mat_op_type const tr = is_complex<T>::value ? mat_herm : mat_trans;
    
    // b should be (n, p)
    test_assert(b.size(0) == n);
    test_assert(b.size(1) == p);
    
    // x should be (n, p)
    test_assert(x.size(0) == n);
    test_assert(x.size(1) == p);
    
    qrd<T, by_reference> qr(m, n, qrd_saveq1);
    
    qr.decompose(a);
    
    Matrix<T> b_1(n, p);
    
    // 1: solve R' b_1 = b
    qr.template rsol<tr>(b, T(1), b_1);
    
    // 2: solve R x = b_1 
    qr.template rsol<mat_ntrans>(b_1, T(1), x);
    
    return x;
  }
};



template <>
struct Covsol_class<by_value>
{

  template <typename T,
	    typename Block0,
	    typename Block1,
	    typename Block2>
  static Matrix<T, Block2>
  covsol(Matrix<T, Block0>       a,
	 const_Matrix<T, Block1> b,
	 Matrix<T, Block2>       x)
  {
    length_type m = a.size(0);
    length_type n = a.size(1);
    length_type p = b.size(1);
    
    mat_op_type const tr = is_complex<T>::value ? mat_herm : mat_trans;
    
    // b should be (n, p)
    test_assert(b.size(0) == n);
    test_assert(b.size(1) == p);
    
    // x should be (n, p)
    test_assert(x.size(0) == n);
    test_assert(x.size(1) == p);
    
    qrd<T, by_value> qr(m, n, qrd_saveq1);
    
    qr.decompose(a);
    
    // 1: solve R' b_1 = b
    Matrix<T> b_1 = qr.template rsol<tr>(b, T(1));
    
    // 2: solve R x = b_1 
    x = qr.template rsol<mat_ntrans>(b_1, T(1));

    Matrix<T> x_chk = qr.covsol(b);

    test_assert(x_chk.size(0) == x.size(0));
    test_assert(x_chk.size(1) == x.size(1));

    for (index_type i=0; i<x.size(0); ++i)
      for (index_type j=0; j<x.size(1); ++j)
	test_assert(equal(x(i,j), x_chk(i,j)));
    
    return x;
  }
};



template <return_mechanism_type RtM,
	  typename              T,
	  typename              Block0,
	  typename              Block1,
	  typename              Block2>
Matrix<T, Block2>
f_covsol(Matrix<T, Block0>       a,
	 const_Matrix<T, Block1> b,
	 Matrix<T, Block2>       x)
{
  return Covsol_class<RtM>::covsol(a, b, x);
}



template <return_mechanism_type RtM,
	  typename              T>
void
test_f_covsol_diag(
  length_type m,
  length_type n,
  length_type p)
{
  test_assert(m >= n);

  Matrix<T> a(m, n);
  Matrix<T> b(n, p);
  Matrix<T> x(n, p);

  // Setup a.
  a        = T();
  a.diag() = T(1);
  if (n > 0) a(0, 0)  = Test_traits<T>::value1();
  if (n > 2) a(2, 2)  = Test_traits<T>::value2();
  if (n > 3) a(3, 3)  = Test_traits<T>::value3();

  // Setup b.
  for (index_type i=0; i<p; ++i)
    test::ramp(b.col(i), T(1), T(i));
  if (p > 1)
    b.col(1) += Test_traits<T>::offset();

  f_covsol<RtM>(a, b, x);

#if VERBOSE
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
#endif

  for (index_type c=0; c<p; ++c)
    for (index_type r=0; r<n; ++r)
      test_assert(equal(b(r, c), tconj<T>(a(r, r)) * a(r, r) * x(r, c)));
}



template <return_mechanism_type RtM,
	  typename              T>
void
test_f_covsol_random(length_type m,
		     length_type n,
		     length_type p)
{
  test_assert(m >= n);

  Matrix<T> a(m, n);
  Matrix<T> b(n, p);
  Matrix<T> x(n, p);

  test::randm(a);
  test::randm(b);

  f_covsol<RtM>(a, b, x);

  Matrix<T> c(n, n);
  Matrix<T> chk(n, p);

  prodh(a, a, c);
  prod(c, x, chk);

  float err = prod_check(c, x, b);

#if VERBOSE
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
  cout << "chk = " << endl << chk << endl;
  cout << "f_covsol<" << Type_name<T>::name()
       << ">(" << m << ", " << n << ", " << p << "): " << err << endl;
#endif

  if (err > 10.0)
  {
    for (index_type r=0; r<n; ++r)
      for (index_type c=0; c<p; ++c)
	test_assert(equal(b(r, c), chk(r, c)));
  }
}



template <return_mechanism_type RtM,
	  typename              T>
void f_covsol_cases(true_type)
{
  test_f_covsol_diag<RtM, T>(1,   1, 2);
  test_f_covsol_diag<RtM, T>(5,   5, 2);
  test_f_covsol_diag<RtM, T>(6,   6, 2);
  test_f_covsol_diag<RtM, T>(17, 17, 2);

  test_f_covsol_diag<RtM, T>(1,   1, 3);
  test_f_covsol_diag<RtM, T>(5,   5, 3);
  test_f_covsol_diag<RtM, T>(17, 17, 3);

  test_f_covsol_diag<RtM, T>(3,   1, 3);
  test_f_covsol_diag<RtM, T>(5,   3, 3);
  test_f_covsol_diag<RtM, T>(17, 11, 3);

  test_f_covsol_random<RtM, T>(1,   1, 2);
  test_f_covsol_random<RtM, T>(5,   5, 2);
  test_f_covsol_random<RtM, T>(17, 17, 2);

  test_f_covsol_random<RtM, T>(1,   1, 3);
  test_f_covsol_random<RtM, T>(5,   5, 3);
  test_f_covsol_random<RtM, T>(17, 17, 3);

  test_f_covsol_random<RtM, T>(3,   1, 3);
  test_f_covsol_random<RtM, T>(5,   3, 3);
  test_f_covsol_random<RtM, T>(17, 11, 3);

#if DO_SWEEP
  for (index_type i=1; i<100; i+= 8)
    for (index_type j=1; j<10; j += 4)
    {
      test_f_covsol_random<RtM, T>(i,   i,   j+1);
      test_f_covsol_random<RtM, T>(i+1, i+1, j);
      test_f_covsol_random<RtM, T>(i+2, i+2, j+2);
    }
#endif
}



template <return_mechanism_type RtM,
	  typename              T>
void f_covsol_cases(false_type)
{
  test_assert(!EXPECT_FULL_COVERAGE);
}



// Front-end function for f_covsol_cases.

template <return_mechanism_type RtM,
	  typename              T>
void f_covsol_cases()
{
  using namespace ovxx::dispatcher;

  f_covsol_cases<RtM, T>(
    integral_constant<bool, is_operation_supported<op::qrd, T>::value>());
}
  


/***********************************************************************
  llsqsol function using rsol tests
***********************************************************************/

// Solve min_x [ norm_2( a x - b ) ]

template <return_mechanism_type ReturnMechanism>
struct Lsqsol_class;

template <>
struct Lsqsol_class<by_reference>
{

  template <typename T,
	    typename Block0,
	    typename Block1,
	    typename Block2>
  static Matrix<T, Block2>
  llsqsol(Matrix<T, Block0>       a,
	  const_Matrix<T, Block1> b,
	  Matrix<T, Block2>       x)
  {
    length_type m = a.size(0);
    length_type n = a.size(1);
    length_type p = b.size(1);


    // b should be (m, p)
    test_assert(b.size(0) == m);
    test_assert(b.size(1) == p);
    
    // x should be (n, p)
    test_assert(x.size(0) == n);
    test_assert(x.size(1) == p);
    
    qrd<T, by_reference> qr(m, n, qrd_saveq);
    
    qr.decompose(a);
    
    mat_op_type const tr = is_complex<T>::value ? mat_herm : mat_trans;
    
    Matrix<T> c(m, p);
    
    // 1. compute C = Q'B:     R X = C
    qr.template prodq<tr, mat_lside>(b, c);
    
    // 2. solve for X:         R X = C
    qr.template rsol<mat_ntrans>(c(Domain<2>(n, p)), T(1), x);
    
    return x;
  }
};



template <>
struct Lsqsol_class<by_value>
{

  template <typename T,
	    typename Block0,
	    typename Block1,
	    typename Block2>
  static Matrix<T, Block2>
  llsqsol(Matrix<T, Block0>       a,
	  const_Matrix<T, Block1> b,
	  Matrix<T, Block2>       x)
  {
    length_type m = a.size(0);
    length_type n = a.size(1);
    length_type p = b.size(1);


    // b should be (m, p)
    test_assert(b.size(0) == m);
    test_assert(b.size(1) == p);
    
    // x should be (n, p)
    test_assert(x.size(0) == n);
    test_assert(x.size(1) == p);
    
    qrd<T, by_value> qr(m, n, qrd_saveq);
    
    qr.decompose(a);
    
    mat_op_type const tr = is_complex<T>::value ? mat_herm : mat_trans;
    
    
    // 1. compute C = Q'B:     R X = C
    Matrix<T> c = qr.template prodq<tr, mat_lside>(b);
    
    // 2. solve for X:         R X = C
    x = qr.template rsol<mat_ntrans>(c(Domain<2>(n, p)), T(1));

    Matrix<T> x_chk = qr.lsqsol(b);

    test_assert(x_chk.size(0) == x.size(0));
    test_assert(x_chk.size(1) == x.size(1));

    for (index_type i=0; i<x.size(0); ++i)
      for (index_type j=0; j<x.size(1); ++j)
	test_assert(equal(x(i,j), x_chk(i,j)));
    
    return x;
  }
};



template <return_mechanism_type RtM,
	  typename              T,
	  typename              Block0,
	  typename              Block1,
	  typename              Block2>
Matrix<T, Block2>
f_llsqsol(Matrix<T, Block0>       a,
	  const_Matrix<T, Block1> b,
	  Matrix<T, Block2>       x)
{
  return Lsqsol_class<RtM>::llsqsol(a, b, x);
}



template <return_mechanism_type RtM,
	  typename              T>
void
test_f_lsqsol_diag(length_type m,
		   length_type n,
		   length_type p)
{
  test_assert(m >= n);

  Matrix<T> a(m, n);
  Matrix<T> x(n, p);
  Matrix<T> b(m, p);

  a        = T();
  a.diag() = T(1);
  if (n > 0) a(0, 0)  = Test_traits<T>::value1();
  if (n > 2) a(2, 2)  = Test_traits<T>::value2();
  if (n > 3) a(3, 3)  = Test_traits<T>::value3();

  for (index_type i=0; i<p; ++i)
    b.col(i) = test::ramp(T(1), T(i), m);
  if (p > 1)
    b.col(1) += Test_traits<T>::offset();

  f_llsqsol<RtM>(a, b, x);

#if VERBOSE
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
#endif

  for (index_type c=0; c<p; ++c)
    for (index_type r=0; r<n; ++r)
      test_assert(equal(b(r, c),
		   a(r, r) * x(r, c)));
}



/// Test QR solver with distributed arguments.

template <return_mechanism_type RtM,
	  typename              T,
	  typename              MapT>
void
test_f_lsqsol_random(length_type m,
		     length_type n,
		     length_type p)
{
  test_assert(m >= n);

  typedef Dense<2, T, row2_type, MapT> block_type;

  MapT map = create_map<2, MapT>();

  Matrix<T, block_type> a(m, n, map);
  Matrix<T, block_type> x(n, p, map);
  Matrix<T, block_type> b(m, p, map);
  Matrix<T, block_type> chk(m, p, map);

  test::randm(a);
  test::randm(b);

  // If m > n, min || AX - B || may not be zero,
  // Need way to check that X is best solution
  //
  // In the meantime, limit rank of A, B to produce zero minimum.

  for (index_type i=n; i<m; ++i)
  {
    // a.row(i) = T(i-n+2) * a.row(i-n);
    // b.row(i) = T(i-n+2) * b.row(i-n);
    a.row(i) = a.row(i-n) * T(i-n+2);
    b.row(i) = b.row(i-n) * T(i-n+2);
  }

  f_llsqsol<RtM>(a, b, x);

  prod(a, x, chk);
  float err = prod_check(a, x, b);

#if VERBOSE
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
  cout << "chk = " << endl << chk << endl;
  cout << "lsqsol<" << Type_name<T>::name()
       << ">(" << m << ", " << n << ", " << p << "): " << err << endl;
#endif

  if (err > 10.0)
  {
    for (index_type r=0; r<m; ++r)
      for (index_type c=0; c<p; ++c)
	test_assert(equal(b(r, c), chk(r, c)));
  }
}



template <return_mechanism_type RtM,
	  typename              T>
void f_lsqsol_cases(true_type)
{
  test_f_lsqsol_diag<RtM, T>(1,   1, 2);
  test_f_lsqsol_diag<RtM, T>(5,   5, 2);
  test_f_lsqsol_diag<RtM, T>(6,   6, 2);
  test_f_lsqsol_diag<RtM, T>(17, 17, 2);

  test_f_lsqsol_diag<RtM, T>(1,   1, 3);
  test_f_lsqsol_diag<RtM, T>(5,   5, 3);
  test_f_lsqsol_diag<RtM, T>(17, 17, 3);

  test_f_lsqsol_diag<RtM, T>(3,   1, 3);
  test_f_lsqsol_diag<RtM, T>(5,   3, 3);
  test_f_lsqsol_diag<RtM, T>(17, 11, 3);

  test_f_lsqsol_random<RtM, T, Local_map>(1,   1, 2);
  test_f_lsqsol_random<RtM, T, Local_map>(5,   5, 2);
  test_f_lsqsol_random<RtM, T, Local_map>(17, 17, 2);

  test_f_lsqsol_random<RtM, T, Local_map>(1,   1, 3);
  test_f_lsqsol_random<RtM, T, Local_map>(5,   5, 3);
  test_f_lsqsol_random<RtM, T, Local_map>(17, 17, 3);

  test_f_lsqsol_random<RtM, T, Local_map>(3,   1, 3);
  test_f_lsqsol_random<RtM, T, Local_map>(5,   3, 3);
  test_f_lsqsol_random<RtM, T, Local_map>(17, 11, 3);

#if DO_SWEEP
  for (index_type i=1; i<100; i+= 8)
    for (index_type j=1; j<10; j += 4)
    {
      test_f_lsqsol_random<RtM, T, Local_map>(i,   i,   j+1);
      test_f_lsqsol_random<RtM, T, Local_map>(i+1, i+1, j);
      test_f_lsqsol_random<RtM, T, Local_map>(i+2, i+2, j+2);
    }
#endif
}



template <return_mechanism_type RtM,
	  typename              T>
void f_lsqsol_cases(false_type)
{
  test_assert(!EXPECT_FULL_COVERAGE);
}



// Front-end function for f_lsqsol_cases.

template <return_mechanism_type RtM,
	  typename              T>
void f_lsqsol_cases()
{
  using namespace ovxx::dispatcher;

  f_lsqsol_cases<RtM, T>(
    integral_constant<bool, is_operation_supported<op::qrd, T>::value>());
}

int
main(int argc, char** argv)
{
  using namespace ovxx::dispatcher;

  vsipl init(argc, argv);

  test::precision<float>::init();
  test::precision<double>::init();

  storage_type st[3];
  st[0] = qrd_nosaveq;
  st[1] = qrd_saveq1;
  st[2] = qrd_saveq;

  for (int i=0; i<3; ++i)
  {
    covsol_cases<float>(st[i]);
    covsol_cases<double>(st[i]);
    covsol_cases<complex<float> >(st[i]);
    covsol_cases<complex<double> >(st[i]);
  }

  for (int i=0; i<3; ++i)
  {
    lsqsol_cases<float>(st[i]);
    lsqsol_cases<double>(st[i]);
    lsqsol_cases<complex<float> >(st[i]);
    lsqsol_cases<complex<double> >(st[i]);
  }

  for (int i=0; i<3; ++i)
  {
    rsol_cases<float>(st[i]);
    rsol_cases<double>(st[i]);
    rsol_cases<complex<float> >(st[i]);
    rsol_cases<complex<double> >(st[i]);
  }

  f_covsol_cases<by_reference, float>();
  f_covsol_cases<by_reference, double>();
  f_covsol_cases<by_reference, complex<float> >();
  f_covsol_cases<by_reference, complex<double> >();

  f_covsol_cases<by_value, float>();
  f_covsol_cases<by_value, double>();
  f_covsol_cases<by_value, complex<float> >();
  f_covsol_cases<by_value, complex<double> >();

  f_lsqsol_cases<by_reference, float>();
  f_lsqsol_cases<by_reference, double>();
  f_lsqsol_cases<by_reference, complex<float> >();
  f_lsqsol_cases<by_reference, complex<double> >();

  f_lsqsol_cases<by_value, float>();
  f_lsqsol_cases<by_value, double>();
  f_lsqsol_cases<by_value, complex<float> >();
  f_lsqsol_cases<by_value, complex<double> >();
#if OVXX_PARALLEL
  test_covsol_random<float, Map<> >(5, 5, 2, qrd_saveq1);
  test_lsqsol_random<float, Map<> >(5, 5, 2, qrd_saveq1);
  test_rsol_diag<float, Map<> >( 5,   5, 2, qrd_saveq1);

  test_f_lsqsol_random<by_reference, float, Map<> >(5,   5, 2);
#endif
}

//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/solvers.hpp>
#include <vsip/map.hpp>
#include <vsip/parallel.hpp>
#include <vsip_csl/diagnostics.hpp>
#include <vsip_csl/load_view.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/test-precision.hpp>

#include <test-random.hpp>
#include "common.hpp"

#define VERBOSE       0
#define DO_ASSERT     1
#define DO_SWEEP      0
#define DO_BIG        1
#define FILE_MATRIX_1 0

#if VERBOSE > 0
#  include <iostream>
#  include <vsip_csl/output.hpp>
#  include <extdata-output.hpp>
#endif

using namespace std;
using namespace vsip;


/***********************************************************************
  Support Definitions
***********************************************************************/

template <typename T,
	  typename Block>
typename vsip::impl::scalar_of<T>::type
norm_2(const_Vector<T, Block> v)
{
  return sqrt(sumval(magsq(v)));
}



/***********************************************************************
  LU tests
***********************************************************************/

double max_err1 = 0.0;
double max_err2 = 0.0;
double max_err3 = 0.0;



// Chold test w/random matrix.


template <typename T,
	  typename Block1,
	  typename Block2>
void
solve_lu(
  return_mechanism_type rtm,
  Matrix<T, Block1>     a,
  Matrix<T, Block2>     b)
{
  length_type n = a.size(0);
  length_type p = b.size(1);

  test_assert(n == a.size(1));
  test_assert(n == b.size(0));

  Matrix<T> x1(n, p);
  Matrix<T> x2(n, p);
  Matrix<T> x3(n, p);

  if (rtm == by_reference)
  {
    // 1. Build solver and factor A.
    lud<T, by_reference> lu(n);
    test_assert(lu.length() == n);

    bool success = lu.decompose(a);
    test_assert(success);

    // 2. Solve A X = B.
    lu.template solve<mat_ntrans>(b, x1);
    lu.template solve<mat_trans>(b, x2);
    lu.template solve<Test_traits<T>::trans>(b, x3); // mat_herm if T complex
  }
  if (rtm == by_value)
  {
    // 1. Build solver and factor A.
    lud<T, by_value> lu(n);
    test_assert(lu.length() == n);

    bool success = lu.decompose(a);
    test_assert(success);

    // 2. Solve A X = B.
    x1 = lu.template solve<mat_ntrans>(b);
    x2 = lu.template solve<mat_trans>(b);
    x3 = lu.template solve<Test_traits<T>::trans>(b); // mat_herm if T complex
  }


  // 3. Check result.

  Matrix<T> chk1(n, p);
  Matrix<T> chk2(n, p);
  Matrix<T> chk3(n, p);

  prod(a, x1, chk1);
  prod(trans(a), x2, chk2);
  prod(trans_or_herm(a), x3, chk3);

  typedef typename vsip::impl::scalar_of<T>::type scalar_type;

  Vector<float> sv_s(n);
  svd<T, by_reference> sv(n, n, svd_uvnos, svd_uvnos);
  sv.decompose(a, sv_s);

  scalar_type a_norm_2 = sv_s(0);


  // Gaussian roundoff error (J.H Wilkinson)
  // (From Moler, Chapter 2.9, p19)
  //
  //  || residual ||
  // ----------------- <= p eps
  // || A || || x_* ||
  //
  // Where 
  //   x_* is computed solution (x is true solution)
  //   residual = b - A x_*
  //   eps is machine precision
  //   p is usually less than 10

  scalar_type eps     = Precision_traits<scalar_type>::eps;
  scalar_type p_limit = scalar_type(20);

#if VERBOSE >= 1
  scalar_type cond = sv_s(0) / sv_s(n-1);
  cout << "solve_lu<" << Type_name<T>::name() << ">("
       << "rtm, "
       << "a = (" << a.size(0) << ", " << a.size(1) << "), "
       << "b = (" << b.size(0) << ", " << b.size(1) << ")):"
       << endl
       << "  a_norm_2 = " << a_norm_2 << endl
       << "  cond     = " << cond << endl
    ;
#endif
  for (index_type i=0; i<p; ++i)
  {
    scalar_type residual_1 = norm_2((b - chk1).col(i));
    scalar_type err1       = residual_1 / (a_norm_2 * norm_2(x1.col(i)) * eps);
    scalar_type residual_2 = norm_2((b - chk2).col(i));
    scalar_type err2       = residual_2 / (a_norm_2 * norm_2(x2.col(i)) * eps);
    scalar_type residual_3 = norm_2((b - chk3).col(i));
    scalar_type err3       = residual_3 / (a_norm_2 * norm_2(x3.col(i)) * eps);

#if VERBOSE == 1
    cout << "  " << i << ": err = "
	 << err1 << ", " << err2 << ", " << err3
	 << endl;
#elif VERBOSE >= 2
    cout << "  " << i << "-1: "
	 << err1 << ", " << residual_1 << ", " << norm_2(x1.col(i)) 
	 << endl;
    cout << "  " << i << "-2: "
	 << err2 << ", " << residual_2 << ", " << norm_2(x2.col(i)) 
	 << endl;
    cout << "  " << i << "-3: "
	 << err3 << ", " << residual_3 << ", " << norm_2(x3.col(i)) 
	 << endl;
#endif

#if DO_ASSERT
    test_assert(err1 < p_limit);
    test_assert(err2 < p_limit);
    test_assert(err3 < p_limit);
#endif

    if (err1 > max_err1) max_err1 = err1;
    if (err2 > max_err2) max_err2 = err2;
    if (err3 > max_err3) max_err3 = err3;
  }

#if VERBOSE >= 3
  cout << "a = " << endl << a << endl;
  cout << "x1 = " << endl << x1 << endl;
  cout << "x2 = " << endl << x2 << endl;
  cout << "b = " << endl << b << endl;
  cout << "chk1 = " << endl << chk1 << endl;
  cout << "chk2 = " << endl << chk2 << endl;
  cout << "chk3 = " << endl << chk3 << endl;
#endif
}



template <typename MapT,
	  typename T,
	  typename Block1,
	  typename Block2>
void
solve_lu_dist(
  return_mechanism_type rtm,
  Matrix<T, Block1>     a,
  Matrix<T, Block2>     b)
{
  length_type n = a.size(0);
  length_type p = b.size(1);

  test_assert(n == a.size(1));
  test_assert(n == b.size(0));

  typedef Dense<2, T, row2_type, MapT> block_type;

  Matrix<T, block_type> x1(n, p);
  Matrix<T, block_type> x2(n, p);
  Matrix<T, block_type> x3(n, p);

  if (rtm == by_reference)
  {
    // 1. Build solver and factor A.
    lud<T, by_reference> lu(n);
    test_assert(lu.length() == n);

    bool success = lu.decompose(a);
    test_assert(success);

    // 2. Solve A X = B.
    lu.template solve<mat_ntrans>(b, x1);
    lu.template solve<mat_trans>(b, x2);
    lu.template solve<Test_traits<T>::trans>(b, x3); // mat_herm if T complex
  }
  if (rtm == by_value)
  {
    // 1. Build solver and factor A.
    lud<T, by_value> lu(n);
    test_assert(lu.length() == n);

    bool success = lu.decompose(a);
    test_assert(success);

    // 2. Solve A X = B.
    impl::assign_local(x1, lu.template solve<mat_ntrans>(b));
    impl::assign_local(x2, lu.template solve<mat_trans>(b));
    impl::assign_local(x3, lu.template solve<Test_traits<T>::trans>(b));
  }


  // 3. Check result.

  Matrix<T, block_type> chk1(n, p);
  Matrix<T, block_type> chk2(n, p);
  Matrix<T, block_type> chk3(n, p);

  prod(a, x1, chk1);
  prod(trans(a), x2, chk2);
  prod(trans_or_herm(a), x3, chk3);

  typedef typename vsip::impl::scalar_of<T>::type scalar_type;

  Vector<float> sv_s(n);
  svd<T, by_reference> sv(n, n, svd_uvnos, svd_uvnos);
  sv.decompose(a, sv_s);

  scalar_type a_norm_2 = sv_s(0);


  // Gaussian roundoff error (J.H Wilkinson)
  // (From Moler, Chapter 2.9, p19)
  //
  //  || residual ||
  // ----------------- <= p eps
  // || A || || x_* ||
  //
  // Where 
  //   x_* is computed solution (x is true solution)
  //   residual = b - A x_*
  //   eps is machine precision
  //   p is usually less than 10

  scalar_type eps     = Precision_traits<scalar_type>::eps;
  scalar_type p_limit = scalar_type(20);

#if VERBOSE >= 1
  scalar_type cond = sv_s(0) / sv_s(n-1);
  cout << "solve_lu_dist<" << Type_name<T>::name() << ">("
       << "rtm, "
       << "a = (" << a.size(0) << ", " << a.size(1) << "), "
       << "b = (" << b.size(0) << ", " << b.size(1) << ")):"
       << endl
       << "  a_norm_2 = " << a_norm_2 << endl
       << "  cond     = " << cond << endl
    ;
#endif

  for (index_type i=0; i<p; ++i)
  {
    scalar_type residual_1 = norm_2((b - chk1).col(i));
    scalar_type err1       = residual_1 / (a_norm_2 * norm_2(x1.col(i)) * eps);
    scalar_type residual_2 = norm_2((b - chk2).col(i));
    scalar_type err2       = residual_2 / (a_norm_2 * norm_2(x2.col(i)) * eps);
    scalar_type residual_3 = norm_2((b - chk3).col(i));
    scalar_type err3       = residual_3 / (a_norm_2 * norm_2(x3.col(i)) * eps);

#if VERBOSE == 1
    cout << "  " << i << ": err = "
	 << err1 << ", " << err2 << ", " << err3
	 << endl;
#elif VERBOSE >= 2
    cout << "  " << i << "-1: "
	 << err1 << ", " << residual_1 << ", " << norm_2(x1.col(i)) 
	 << endl;
    cout << "  " << i << "-2: "
	 << err2 << ", " << residual_2 << ", " << norm_2(x2.col(i)) 
	 << endl;
    cout << "  " << i << "-3: "
	 << err3 << ", " << residual_3 << ", " << norm_2(x3.col(i)) 
	 << endl;
#endif

    test_assert(err1 < p_limit);
    test_assert(err2 < p_limit);
    test_assert(err3 < p_limit);

    if (err1 > max_err1) max_err1 = err1;
    if (err2 > max_err2) max_err2 = err2;
    if (err3 > max_err3) max_err3 = err3;
  }

#if VERBOSE >= 3
  cout << "a = " << endl << a << endl;
  cout << "x1 = " << endl << x1 << endl;
  cout << "x2 = " << endl << x2 << endl;
  cout << "b = " << endl << b << endl;
  cout << "chk1 = " << endl << chk1 << endl;
  cout << "chk2 = " << endl << chk2 << endl;
  cout << "chk3 = " << endl << chk3 << endl;
#endif
}



// Simple lud test w/diagonal matrix.

template <typename T>
void
test_lud_diag(
  return_mechanism_type rtm,
  length_type n,
  length_type p)
{
  Matrix<T> a(n, n);
  Matrix<T> b(n, p);

  a        = T();
  a.diag() = T(1);
  if (n > 0) a(0, 0)  = mag(Test_traits<T>::value1());
  if (n > 2) a(2, 2)  = mag(Test_traits<T>::value2());
  if (n > 3) a(3, 3)  = mag(Test_traits<T>::value3());

  for (index_type i=0; i<p; ++i)
    b.col(i) = test_ramp(T(1), T(i), n);
  if (p > 1)
    b.col(1) += Test_traits<T>::offset();

  solve_lu(rtm, a, b);
}



// Chold test w/random matrix.

template <typename T>
void
test_lud_random(
  return_mechanism_type rtm,
  length_type           n,
  length_type           p)
{
  Matrix<T> a(n, n);
  Matrix<T> b(n, p);

  randm(a);
  randm(b);

  solve_lu(rtm, a, b);
}



// Chold test w/random matrix.

template <typename T,
	  typename MapA,
	  typename MapB,
	  typename MapT>
void
test_lud_dist(
  return_mechanism_type rtm,
  length_type           n,
  length_type           p)
{
  typedef Dense<2, T, row2_type, MapA> a_block_type;
  typedef Dense<2, T, row2_type, MapB> b_block_type;

  Matrix<T, a_block_type> a(n, n);
  Matrix<T, b_block_type> b(n, p);

  randm(a);
  randm(b);

  solve_lu_dist<MapT>(rtm, a, b);
}



// Chold test w/matrix from file.

template <typename FileT,
	  typename T>
void
test_lud_file(
  return_mechanism_type rtm,
  char*                 afilename,
  char*                 bfilename,
  length_type           n,
  length_type           p)
{
  vsip_csl::Load_view<2, FileT> load_a(afilename, Domain<2>(n, n));
  vsip_csl::Load_view<2, FileT> load_b(bfilename, Domain<2>(n, p));

  Matrix<T> a(n, n);
  Matrix<T> b(n, p);

  a = load_a.view();
  b = load_b.view();

  solve_lu(rtm, a, b);
}



// Run LU tests when type T is supported.
// Called by lud_cases front-end function below.

template <typename T>
void lud_cases(return_mechanism_type rtm, vsip::impl::integral_constant<bool, true>)
{
  for (index_type p=1; p<=3; ++p)
  {
    test_lud_diag<T>(rtm, 1, p);
    test_lud_diag<T>(rtm, 5, p);
    test_lud_diag<T>(rtm, 6, p);
    test_lud_diag<T>(rtm, 17, p);
  }


  for (index_type p=1; p<=3; ++p)
  {
    test_lud_random<T>(rtm, 1, p);
    test_lud_random<T>(rtm, 2, p);
    test_lud_random<T>(rtm, 5, p);
    test_lud_random<T>(rtm, 6, p);
    test_lud_random<T>(rtm, 16, p);
    test_lud_random<T>(rtm, 17, p);
  }

#if DO_BIG
  test_lud_random<T>(rtm, 97,   5+1);
  test_lud_random<T>(rtm, 97+1, 5);
  test_lud_random<T>(rtm, 97+2, 5+2);
#endif

#if DO_SWEEP
  for (index_type i=1; i<100; i+= 8)
    for (index_type j=1; j<10; j += 4)
    {
      test_lud_random<T>(rtm, i,   j+1);
      test_lud_random<T>(rtm, i+1, j);
      test_lud_random<T>(rtm, i+2, j+2);
    }
#endif
}



// Don't run LU tests when type T is not supported.
// Called by lud_cases front-end function below.

template <typename T>
void lud_cases(return_mechanism_type, vsip::impl::integral_constant<bool, false>)
{
  // std::cout << "lud_cases " << Type_name<T>::name() << " not supported\n";
}



// Front-end function for lud_cases.

// This function dispatches to either real set of tests or an empty
// function depending on whether the LU backends configured in support
// value type T.  (Not all LU backends support all value types).

template <typename T>
void lud_cases(return_mechanism_type rtm)
{
  using vsip::impl::integral_constant;
  using namespace vsip_csl::dispatcher;
  lud_cases<T>(rtm,
	       integral_constant<bool,
	       is_operation_supported<op::lud, T>::value &&
	       is_operation_supported<op::svd, T>::value>());
}



template <typename T>
void
dist_lud_cases()
{
  typedef Map<Block_dist, Block_dist> map1_type;
  typedef Map<Block_dist, Block_dist> map2_type;
  typedef Map<Block_dist, Block_dist> map3_type;

  test_lud_dist<T, map1_type, map2_type, map3_type>(by_reference, 5, 7);
  test_lud_dist<T, map1_type, map2_type, map3_type>(by_value,     5, 7);
}



/***********************************************************************
  Main
***********************************************************************/

template <> float  Precision_traits<float>::eps = 0.0;
template <> double Precision_traits<double>::eps = 0.0;



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  Precision_traits<float>::compute_eps();
  Precision_traits<double>::compute_eps();

#if VERBOSE >= 1
  std::cout << "Precision_traits<float>::eps = "
	    << Precision_traits<float>::eps 
	    << std::endl;
  std::cout << "Precision_traits<double>::eps = "
	    << Precision_traits<double>::eps 
	    << std::endl;
#endif

#if FILE_MATRIX_1
  test_lud_file<complex<float>, complex<double> >(
    "lu-a-complex-float-99x99.dat", "lu-b-complex-float-99x7.dat", 99, 7);
  test_lud_file<complex<float>, complex<float> >(
    "lu-a-complex-float-99x99.dat", "lu-b-complex-float-99x7.dat", 99, 7);
#endif

  test_lud_diag<complex<float> >(by_reference, 17, 3);

  lud_cases<float>           (by_reference);
  lud_cases<double>          (by_reference);
  lud_cases<complex<float> > (by_reference);
  lud_cases<complex<double> >(by_reference);

  lud_cases<float>           (by_value);
  lud_cases<double>          (by_value);
  lud_cases<complex<float> > (by_value);
  lud_cases<complex<double> >(by_value);

  dist_lud_cases<float>      ();
}

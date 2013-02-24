/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/solvers/cholesky/cholesky.cpp
    @author  Jules Bergmann
    @date    2005-08-19
    @brief   VSIPL++ Library: Common routines for cholesky solver unit tests.
*/

#ifndef TESTS_SOLVERS_CHOLESKY_CHOLESKY_HPP
#define TESTS_SOLVERS_CHOLESKY_CHOLESKY_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/solvers.hpp>
#include <vsip_csl/diagnostics.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/test-precision.hpp>
#include <test-random.hpp>
#include <solvers/common.hpp>

#define VERBOSE      0
#define DO_SWEEP     0
#define BAD_MATRIX_A 0

#if VERBOSE
#  include <iostream>
#  include <vsip_csl/output.hpp>
#  include <extdata-output.hpp>
#endif

enum chold_what
{
  base_test,
  big_test,
  sweep_test
};

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Load_view utility.
***********************************************************************/

// This is nearly same as sarsim LoadView, but doesn't include byte
// order.  Move this into common location.

template <typename T>
struct Load_view_traits
{
  typedef T base_t;
  static unsigned const factor = 1;
};

template <typename T>
struct Load_view_traits<vsip::complex<T> >
{
  typedef T base_t;
  static unsigned const factor = 2;
};


template <vsip::dimension_type Dim,
	  typename             T>
class Load_view
{
public:
  typedef typename Load_view_traits<T>::base_t base_t;
  static unsigned const factor = Load_view_traits<T>::factor;

  typedef vsip::Dense<Dim, T> block_t;
  typedef typename vsip::impl::view_of<block_t>::type view_t;

public:
  Load_view(char*                    filename,
	    vsip::Domain<Dim> const& dom)
    : data_  (new base_t[factor*dom.size()]),
      block_ (dom, data_),
      view_  (block_)
  {
    FILE*  fd;
    size_t size = dom.size();
    
    if (!(fd = fopen(filename,"r")))
    {
      fprintf(stderr,"Load_view: error opening '%s'.\n", filename);
      exit(1);
    }

    if (size != fread(data_, sizeof(T), size, fd))
    {
      fprintf(stderr, "Load_view: error reading file %s.\n", filename);
      exit(1);
    }
  
    fclose(fd);
    
    block_.admit(true);
  }



  Load_view(FILE*              fd,
	    vsip::Domain<Dim> const& dom)
    : data_  (new base_t[factor*dom.size()]),
      block_ (dom, data_),
      view_  (block_)
  {
    size_t size = dom.size();

    if (size != fread(data_, sizeof(T), size, fd))
    {
      fprintf(stderr, "Load_view: error reading file.\n");
      exit(1);
    }
    
    block_.admit(true);
  }

  ~Load_view()
  { delete[] data_; }

  view_t view() { return view_; }

private:
  base_t*       data_;

  block_t       block_;
  view_t        view_;
};



/***********************************************************************
  Covsol tests
***********************************************************************/

// Simple chold test w/diagonal matrix.

template <typename T>
void
test_chold_diag(
  mat_uplo    uplo,
  length_type n,
  length_type p)
{
  Matrix<T> a(n, n);
  Matrix<T> b(n, p);
  Matrix<T> x(n, p);

  // For a diagonal matrix to be positive-definite, all diagonal elements
  // must be (a) real and (b) > 0.

  a        = T();
  a.diag() = T(1);
  if (n > 0) a(0, 0)  = mag(Test_traits<T>::value1());
  if (n > 2) a(2, 2)  = mag(Test_traits<T>::value2());
  if (n > 3) a(3, 3)  = mag(Test_traits<T>::value3());

  chold<T, by_reference> chol(uplo, n);

  test_assert(chol.uplo()   == uplo);
  test_assert(chol.length() == n);

  bool success = chol.decompose(a);

  test_assert(success);

  for (index_type i=0; i<p; ++i)
    b.col(i) = test_ramp(T(1), T(i), n);
  if (p > 1)
    b.col(1) += Test_traits<T>::offset();

  chol.solve(b, x);

#if VERBOSE
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
#endif

  for (index_type c=0; c<p; ++c)
    for (index_type r=0; r<n; ++r)
      test_assert(equal(b(r, c), a(r, r) * x(r, c)));
}



// Chold test w/random matrix.

template <typename T>
void
test_chold_random(
  mat_uplo    uplo,
  length_type n,
  length_type p)
{
  Matrix<T> a(n, n);
  Matrix<T> w(n, n);
  Matrix<T> b(n, p);
  Matrix<T> x(n, p);

  // 1. Construct a symmetric/hermetian positive-definite A by "cross-product"
  //     - A = W' W

  randm(w);

  prodh(w, w, a);

  // Check A symmetric/hermetian.
  for (index_type i=0; i<n; ++i)
  {
    test_assert(is_positive<T>(a(i, i)));
    for (index_type j=0; j<i; ++j)
      test_assert(equal(a(i, j), tconj<T>(a(j, i))));
  }
  


  // 2. Build solver and factor A.

  chold<T, by_reference> chol(uplo, n);

  test_assert(chol.uplo()   == uplo);
  test_assert(chol.length() == n);

  bool success = chol.decompose(a);

  test_assert(success);


  // 3. Solve A X = B.

  randm(b);

  chol.solve(b, x);


  // 4. Check result.

  Matrix<T> chk(n, p);

  prod(a, x, chk);

  float err = prod_check(a, x, b);

#if VERBOSE
  cout << "w = " << endl << w << endl;
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
  cout << "chk = " << endl << chk << endl;
  cout << "covsol<" << Type_name<T>::name()
       << ">(" << n << ", " << ", " << p << "): " << err << endl;
#endif

  if (err > 10.0)
  {
    for (index_type r=0; r<n; ++r)
      for (index_type c=0; c<p; ++c)
	test_assert(equal(b(r, c), chk(r, c)));
  }
}



// Chold test w/matrix from file.

template <typename T>
void
test_chold_file(
  mat_uplo    uplo,
  char*       filename,
  length_type n,
  length_type p)
{
  Load_view<2, T> load_a(filename, Domain<2>(n, n));

  Matrix<T> a(n, n);
  Matrix<T> b(n, p);
  Matrix<T> x(n, p);

  // 1. Construct a symmetric/hermetian positive-definite A by "cross-product"
  //     - A = W' W

  a = load_a.view();


  // Check A symmetric/hermetian.
  for (index_type i=0; i<n; ++i)
  {
    test_assert(is_positive<T>(a(i, i)));
    for (index_type j=0; j<i; ++j)
      test_assert(equal(a(i, j), tconj<T>(a(j, i))));
  }
  


  // 2. Build solver and factor A.

  chold<T, by_reference> chol(uplo, n);

  test_assert(chol.uplo()   == uplo);
  test_assert(chol.length() == n);

  bool success = chol.decompose(a);

  test_assert(success);



  // 3. Solve A X = B.

  randm(b);

  chol.solve(b, x);


  // 4. Check result.

  Matrix<T> chk(n, p);

  prod(a, x, chk);

  float err = prod_check(a, x, b);

#if VERBOSE
  cout << "a = " << endl << a << endl;
  cout << "x = " << endl << x << endl;
  cout << "b = " << endl << b << endl;
  cout << "chk = " << endl << chk << endl;
  cout << "covsol<" << Type_name<T>::name()
       << ">(" << n << ", " << ", " << p << "): " << err << endl;
#endif

  if (err > 10.0)
  {
    for (index_type r=0; r<n; ++r)
      for (index_type c=0; c<p; ++c)
	test_assert(equal(b(r, c), chk(r, c)));
  }
}



/***********************************************************************
  Chold cases
***********************************************************************/

// Run Chold tests when type T is supported.
// Called by chold_cases front-end function below.

template <typename T>
void chold_cases(mat_uplo uplo, vsip::impl::true_type)
{
  for (index_type p=1; p<=3; ++p)
  {
    test_chold_diag<T>(uplo,  1, p);
    test_chold_diag<T>(uplo,  5, p);
    test_chold_diag<T>(uplo,  6, p);
    test_chold_diag<T>(uplo, 17, p);
  }


  for (index_type p=1; p<=3; ++p)
  {
    test_chold_random<T>(uplo,  1, p);
    test_chold_random<T>(uplo,  2, p);
    test_chold_random<T>(uplo,  5, p);
    test_chold_random<T>(uplo,  6, p);
    test_chold_random<T>(uplo, 16, p);
    test_chold_random<T>(uplo, 17, p);
  }

#if DO_SWEEP
  for (index_type i=1; i<100; i+= 8)
    for (index_type j=1; j<10; j += 4)
    {
      test_chold_random<T>(uplo, i,   j+1);
      test_chold_random<T>(uplo, i+1, j);
      test_chold_random<T>(uplo, i+2, j+2);
    }
#endif
}



// Don't run Chold tests when type T is not supported.
// Called by chold_cases front-end function below.

template <typename T>
void chold_cases(mat_uplo, vsip::impl::false_type)
{
  // std::cout << "chold_cases " << Type_name<T>::name() << " not supported\n";
}



// Front-end function for chold_cases.

// This function dispatches to either real set of tests or an empty
// function depending on whether the Chold backends configured in support
// value type T.  (Not all Chold backends support all value types).

template <typename T>
void chold_cases(mat_uplo uplo)
{
  using vsip::impl::integral_constant;
  using namespace vsip_csl::dispatcher;

  chold_cases<T>(uplo,
    integral_constant<bool, is_operation_supported<op::chold, T>::value>());
}



/***********************************************************************
  Chold big cases
***********************************************************************/

// Run Chold tests when type T is supported.
// Called by chold_cases front-end function below.

template <typename T>
void chold_big_cases(mat_uplo uplo, vsip::impl::true_type)
{
  test_chold_random<T>(uplo, 97,   5+1);
  test_chold_random<T>(uplo, 97+1, 5);
  test_chold_random<T>(uplo, 97+2, 5+2);
}



// Don't run Chold tests when type T is not supported.
// Called by chold_cases front-end function below.

template <typename T>
void chold_big_cases(mat_uplo, vsip::impl::false_type)
{
  // std::cout << "chold_cases " << Type_name<T>::name() << " not supported\n";
}



// Front-end function for chold_cases.

// This function dispatches to either real set of tests or an empty
// function depending on whether the Chold backends configured in support
// value type T.  (Not all Chold backends support all value types).

template <typename T>
void chold_big_cases(mat_uplo uplo)
{
  using vsip::impl::integral_constant;
  using namespace vsip_csl::dispatcher;

  chold_big_cases<T>(uplo,
    integral_constant<bool, is_operation_supported<op::chold, T>::value>());
}

#endif // TESTS_SOLVERS_CHOLESKY_CHOLESKY_HPP

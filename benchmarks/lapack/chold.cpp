/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for Cholesky Decomposition and linear system solver.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <vsip/random.hpp>
#include <vsip/math.hpp>
#include <vsip/solvers.hpp>

#include <vsip_csl/profile.hpp>
#include <vsip_csl/test.hpp>
#include "loop.hpp"
#include "common.hpp"

using namespace vsip;

// Decomposition benchmark
template <typename T>
struct t_chold : Benchmark_base
{
  char const* what() { return "t_chold"; }
  float ops_per_point(length_type n)
  {
    float ops;

    ops = n * n * n / 3.f;

    return impl::is_complex<T>::value ? (4.f * ops / n) : (ops / n);
  }

  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type size) { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef typename impl::scalar_of<T>::type scalar_type;

    Matrix<T> A(size, size, T());
    Matrix<T> A_saved(size, size, T());

    randm_symm(A_saved);

    chold<T, by_reference> chol(uplo_, size);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
    {
      A = A_saved;
      chol.decompose(A);
    }
    t1.stop();
    
    time = t1.delta();
  }

  t_chold(mat_uplo uplo) : uplo_(uplo) {}

  mat_uplo uplo_;
};

// Solve with by-reference return mechanism
template <typename T, mat_op_type tr>
struct t_chold_solve_br : Benchmark_base
{
  char const* what() { return "t_chold_solve_br"; }
  float ops_per_point(length_type n)
  {
    float ops;

    ops = 2.f * n * n * b_ncols_;

    return impl::is_complex<T>::value ? (4.f * ops / n) : (ops / n);
  }

  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type size) { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Matrix<T> A(size, size, T());
    Matrix<T> B(size, b_ncols_, T());
    Matrix<T> X(size, b_ncols_, T());

    randm_symm(A);
    randm(B);

    chold<T, by_reference> chol(uplo_, size);
    
    chol.decompose(A);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
      chol.solve(B, X);
    t1.stop();
    
    time = t1.delta();
  }

  t_chold_solve_br(length_type b_ncols, mat_uplo uplo)
  : b_ncols_(b_ncols),
    uplo_(uplo) {}

  length_type b_ncols_;
  mat_uplo uplo_;
};

// Solve with by-value return mechanism
template <typename T, mat_op_type tr>
struct t_chold_solve_bv : Benchmark_base
{
  char const* what() { return "t_chold_solve_bv"; }
  float ops_per_point(length_type n)
  {
    float ops;

    ops = 2.f * n * n * b_ncols_;

    return impl::is_complex<T>::value ? (4.f * ops / n) : (ops / n);
  }

  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type size) { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef typename impl::scalar_of<T>::type scalar_type;

    Matrix<T> A(size, size, T());
    Matrix<T> B(size, b_ncols_, T());
    Matrix<T> X(size, b_ncols_, T());

    randm_symm(A);
    randm(B);

    chold<T, by_value> chol(uplo_, size);
    
    chol.decompose(A);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
      X = chol.solve(B);
    t1.stop();
    
    time = t1.delta();
  }

  t_chold_solve_bv(length_type b_ncols, mat_uplo uplo)
  : b_ncols_(b_ncols),
    uplo_(uplo) {}

  length_type b_ncols_;
  mat_uplo uplo_;
};

void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;
  loop.stop_ = 12;
  loop.user_param_ = 3;
}



int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1 : loop(t_chold_solve_br<float, mat_ntrans>(loop.user_param_, upper)); break;
  case  2 : loop(t_chold_solve_br<std::complex<float>, mat_ntrans>(loop.user_param_, upper)); break;
  case  11: loop(t_chold_solve_br<float, mat_trans>(loop.user_param_, lower)); break;
  case  21: loop(t_chold_solve_br<std::complex<float>, mat_trans>(loop.user_param_, lower)); break;

  case  12: loop(t_chold_solve_bv<float, mat_ntrans>(loop.user_param_, upper)); break;
  case  22: loop(t_chold_solve_bv<std::complex<float>, mat_ntrans>(loop.user_param_, upper)); break;
  case  13: loop(t_chold_solve_bv<float, mat_trans>(loop.user_param_, lower)); break;
  case  23: loop(t_chold_solve_bv<std::complex<float>, mat_trans>(loop.user_param_, lower)); break;

  case  14: loop(t_chold<float>(upper)); break;
  case  24: loop(t_chold<std::complex<float> >(upper)); break;
  case  15: loop(t_chold<float>(lower)); break;
  case  25: loop(t_chold<std::complex<float> >(lower)); break;

  case 0:
    std::cout
      << "chold -- Cholesky Decomposition and Solver.\n"
      << " Solve:\n"
      << "  -1 : by reference, float,   upper triangular decomposition\n"
      << "  -2 : by reference, complex, upper triangular decomposition\n"
      << "  -11: by reference, float,   lower triangular decomposition\n"
      << "  -21: by reference, complex, lower triangular decomposition\n"
      << "\n"
      << "  -12: by     value, float,   upper triangular decomposition\n"
      << "  -22: by     value, complex, upper triangular decomposition\n"
      << "  -13: by     value, float,   lower triangular decomposition\n"
      << "  -23: by     value, complex, lower triangular decomposition\n"
      << "\n"
      << " Decompose only:\n"
      << "  -14: float,    upper triangular decomposition\n"
      << "  -24: complex,  upper triangular decomposition\n"
      << "  -15: float,    lower triangular decomposition\n"
      << "  -25: complex,  lower triangular decomposition\n"
      << "\n"
      << " Parameters:\n"
      << "  -param N      -- number of columns in solution matrix (default 3)\n"
      ;
  default: return 0;
  }
  return 1;
}

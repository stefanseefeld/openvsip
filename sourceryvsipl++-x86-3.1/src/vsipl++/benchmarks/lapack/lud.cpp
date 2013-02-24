/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for LU Decomposition and linear system solver.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/solvers.hpp>

#include <vsip_csl/profile.hpp>
#include <vsip_csl/test.hpp>
#include "loop.hpp"
#include "common.hpp"

using namespace vsip;

// Decomposition benchmark
template <typename T>
struct t_lud : Benchmark_base
{
  char const* what() { return "t_lud"; }
  float ops_per_point(length_type n)
  {
    float ops;

    ops = 2.f * n * n * n / 3.f;

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

    randm(A_saved);

    lud<T, by_reference> lu(size);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
    {
      A = A_saved;
      lu.decompose(A);
    }
    t1.stop();
    
    time = t1.delta();
  }

  t_lud() {}
};

// Solve with by-reference return mechanism
template <typename T, mat_op_type tr>
struct t_lud_solve_br : Benchmark_base
{
  char const* what() { return "t_lud_solve_br"; }
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

    randm(A);
    randm(B);

    lud<T, by_reference> lu(size);
    
    lu.decompose(A);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
      lu.template solve<tr>(B, X);
    t1.stop();
    
    time = t1.delta();
  }

  t_lud_solve_br(length_type b_ncols) : b_ncols_(b_ncols) {}

  length_type b_ncols_;

};

// Solve with by-value return mechanism
template <typename T, mat_op_type tr>
struct t_lud_solve_bv : Benchmark_base
{
  char const* what() { return "t_lud_solve_bv"; }
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

    randm(A);
    randm(B);

    lud<T, by_value> lu(size);
    
    lu.decompose(A);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
      X = lu.template solve<tr>(B);
    t1.stop();
    
    time = t1.delta();
  }

  t_lud_solve_bv(length_type b_ncols) : b_ncols_(b_ncols) {}

  length_type b_ncols_;
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
  case  1 : loop(t_lud_solve_br<float, mat_ntrans>(loop.user_param_)); break;
  case  2 : loop(t_lud_solve_br<std::complex<float>, mat_ntrans>(loop.user_param_)); break;
  case  11: loop(t_lud_solve_br<float, mat_trans>(loop.user_param_)); break;
  case  21: loop(t_lud_solve_br<std::complex<float>, mat_trans>(loop.user_param_)); break;
  case  20: loop(t_lud_solve_br<std::complex<float>, mat_herm>(loop.user_param_)); break;

  case  12: loop(t_lud_solve_bv<float, mat_ntrans>(loop.user_param_)); break;
  case  22: loop(t_lud_solve_bv<std::complex<float>, mat_ntrans>(loop.user_param_)); break;
  case  13: loop(t_lud_solve_bv<float, mat_trans>(loop.user_param_)); break;
  case  23: loop(t_lud_solve_bv<std::complex<float>, mat_trans>(loop.user_param_)); break;
  case  29: loop(t_lud_solve_bv<std::complex<float>, mat_herm>(loop.user_param_)); break;

  case  14: loop(t_lud<float>()); break;
  case  24: loop(t_lud<std::complex<float> >()); break;

  case 0:
    std::cout
      << "lud -- LU Decomposition and Solver.\n"
      << "\n"
      << " Solve:\n"
      << "  -1 : by reference, float,   op = mat_ntrans\n"
      << "  -2 : by reference, complex, op = mat_ntrans\n"
      << "  -11: by reference, float,   op = mat_trans\n"
      << "  -21: by reference, complex, op = mat_trans\n"
      << "  -20: by reference, complex, op = mat_herm\n"
      << "\n"
      << "  -12: by     value, float,   op = mat_ntrans\n"
      << "  -22: by     value, complex, op = mat_ntrans\n"
      << "  -13: by     value, float,   op = mat_trans\n"
      << "  -23: by     value, complex, op = mat_trans\n"
      << "  -29: by     value, complex, op = mat_herm\n"
      << " Decompose:\n"
      << "   -14: float\n"
      << "   -24: complex\n"
      << "\n"
      << " Parameters:\n"
      << "  -param N      -- number of columns in solution matrix (default 3)\n"
      ;
  default: return 0;
  }
  return 1;
}

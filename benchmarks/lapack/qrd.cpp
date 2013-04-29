//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for QR factorization.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/solvers.hpp>

#include <vsip_csl/profile.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"
#include "common.hpp"

using namespace vsip;

#define HAVE_LAPACK 0

#if HAVE_LAPACK

// This is not a Lapack QR benchmark, rather it passes addition
// flags to the Lapack backend to control whether the QR is blocked
// (for large matrix performance) or not (for small matrix performance).

template <typename T,
	  bool     Blocked>
struct t_qrd_lapack1
{
  char const* what() { return "t_qrd1"; }
  float ops_per_point(length_type n)
  {
    return impl::is_complex<T>::value ? (16.f * n * n /* n */ / 3)
	                              : ( 4.f * n * n /* n */ / 3);
  }

  int riob_per_point(length_type) { return -1*sizeof(T); }
  int wiob_per_point(length_type) { return -1*sizeof(T); }
  int mem_per_point(length_type size) { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Matrix<T>   A(size, size, T());


    // A = T(); A.diag() = T(1);
    randm(A);

    impl::lapack::Qrd<T, Blocked> qr(size, size, qrd_saveq);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
      qr.decompose(A);
    t1.stop();
    
    time = t1.delta();
  }
};
#endif


template <typename T,
	  typename OrderT>
struct t_qrd : Benchmark_base
{
  char const* what() { return "t_qrd"; }
  float ops_per_point(length_type n)
  {
    length_type m = ratio_ * n;
    float ops = impl::is_complex<T>::value ? (8.f * n * n * (3*m - n) / 3)
	                                   : (2.f * n * n * (3*m - n) / 3);
    return ops/n;
  }

  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type size) { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Dense<2, T, OrderT> block_type;
    Matrix<T, block_type>   A(ratio_*size, size, T());

    // A = T(); A.diag() = T(1);
    randm(A);

    qrd<T, by_reference> qr(ratio_*size, size, st_);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
      qr.decompose(A);
    t1.stop();
    
    time = t1.delta();
  }

  t_qrd(length_type ratio, storage_type st) : ratio_(ratio), st_(st) {}

  length_type  ratio_;
  storage_type st_;

};

// Solve benchmark
template <typename T, mat_op_type tr>
struct t_qrd_solve : Benchmark_base
{
  char const* what() { return "t_qrd_solve"; }
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
    T alpha = T(1);

    randm(A);
    randm(B);

    qrd<T, by_reference> qr(size, size, st_);
    
    qr.decompose(A);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
      qr.template rsol<tr>(B, alpha, X);
    t1.stop();
    
    time = t1.delta();
  }

  t_qrd_solve(storage_type st, length_type b_ncols) : st_(st), b_ncols_(b_ncols) {}

  storage_type st_;
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
  case  1: loop(t_qrd<complex<float>, row2_type >(3, qrd_nosaveq)); break;
  case  2: loop(t_qrd<complex<float>, row2_type >(3, qrd_saveq1)); break;
  case  3: loop(t_qrd<complex<float>, row2_type >(3, qrd_saveq)); break;

  case 11: loop(t_qrd<complex<float>, col2_type >(3, qrd_nosaveq)); break;
  case 12: loop(t_qrd<complex<float>, col2_type >(3, qrd_saveq1)); break;
  case 13: loop(t_qrd<complex<float>, col2_type >(3, qrd_saveq)); break;

#if HAVE_LAPACK
  // These can be enabled if using Lapack as a backend.
  case  11: loop(t_qrd_lapack1<complex<float>, true>()); break;
  case  12: loop(t_qrd_lapack1<complex<float>, false>()); break;
#endif
  case 21: loop(t_qrd_solve<float, mat_ntrans>(qrd_saveq, loop.user_param_)); break;
  case 22: loop(t_qrd_solve<complex<float>, mat_ntrans>(qrd_saveq, loop.user_param_)); break;
  case 23: loop(t_qrd_solve<float, mat_trans>(qrd_saveq, loop.user_param_)); break;
  case 24: loop(t_qrd_solve<complex<float>, mat_trans>(qrd_saveq, loop.user_param_)); break;
  case 25: loop(t_qrd_solve<complex<float>, mat_herm>(qrd_saveq, loop.user_param_)); break;

  case 0:
    std::cout
      << "qrd -- QR factorization.\n"
      << "   -1: by    row, storage=qrd_nosaveq\n"
      << "   -2: by    row, storage=qrd_saveq1\n"
      << "   -3: by    row, storage=qrd_saveq\n"
      << "\n"
      << "  -11: by column, storage=qrd_nosaveq\n"
      << "  -12: by column, storage=qrd_saveq1\n"
      << "  -13: by column, storage=qrd_saveq\n"
      << "\n"
      << " op(R) Solve:\n"
      << "  -21: float,   op = mat_ntrans\n"
      << "  -22: complex, op = mat_ntrans\n"
      << "  -23: float,   op = mat_trans\n"
      << "  -24: complex, op = mat_trans\n"
      << "  -25: complex, op = mat_herm\n"
      << "\n"
      << " Parameters:\n"
      << "  -param N      -- number of columns in solution matrix (default 3)\n"
      ;
  default: return 0;
  }
  return 1;
}

//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for SVD of square matrices.

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

// Decomposition with by-reference return mechanism
template <typename T>
struct t_svd_br : Benchmark_base
{
  char const* what() { return "t_svd_br"; }
  float ops_per_point(length_type n)
  {
    float ops;

    // FLOP count for LAPACK DGESVD benchmark in the LAPACK Users Guide
    if (st_ == svd_uvnos)
      ops = 2.67 * n * n * n;
    else
      ops = 6.67 * n * n * n;

    return impl::is_complex<T>::value ? (4.f * ops / n) : (ops / n);
  }

  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type size) { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef typename impl::scalar_of<T>::type scalar_type;

    Matrix<T>           A(size, size, T());
    Vector<scalar_type> V(size);

    randm(A);

    svd<T, by_reference> sv(size, size, st_, st_);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
      sv.decompose(A, V);
    t1.stop();
    
    time = t1.delta();
  }

  t_svd_br(storage_type st) : st_(st) {}

  storage_type st_;
};

// Decomposition with by-value return mechanism
template <typename T>
struct t_svd_bv : Benchmark_base
{
  char const* what() { return "t_svd_bv"; }
  float ops_per_point(length_type n)
  {
    float ops;
    // FLOP count for LAPACK DGESVD benchmark in the LAPACK Users Guide
    if (st_ == svd_uvnos)
      ops = 2.67 * n * n * n;
    else
      ops = 6.67 * n * n * n;

    return impl::is_complex<T>::value ? (4.f * ops / n) : (ops / n);
  }

  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type size) { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef typename impl::scalar_of<T>::type scalar_type;

    Matrix<T>           A(size, size, T());
    Vector<scalar_type> V(size);

    randm(A);

    svd<T, by_value> sv(size, size, st_, st_);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
      V = sv.decompose(A);
    t1.stop();
    
    time = t1.delta();
  }

  t_svd_bv(storage_type st) : st_(st) {}

  storage_type st_;
};

// Decomposition and product with by-reference return mechanism
template <typename T>
struct t_svd_produ_br : Benchmark_base
{
  char const* what() { return "t_svd_produ_br"; }
  float ops_per_point(length_type n)
  {
    float ops;
    // FLOP count for LAPACK DGESVD benchmark in the LAPACK Users Guide
    //  with matrix multiplication
    if (st_ == svd_uvnos)
      ops = 4.67 * n * n * n - n * n;
    else
      ops = 8.67 * n * n * n - n * n;

    return impl::is_complex<T>::value ? (4.f * ops / n) : (ops / n);
  }

  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type size) { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef typename impl::scalar_of<T>::type scalar_type;

    Matrix<T>           A(size, size, T());
    Matrix<T>           C(size, size, T());
    Matrix<T>           X(size, size);
    Vector<scalar_type> V(size);

    randm(A);
    randm(C);

    svd<T, by_reference> sv(size, size, st_, st_);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
    {
      sv.decompose(A, V);
      sv.template produ<mat_ntrans, mat_lside>(C, X);
    }
    t1.stop();
    
    time = t1.delta();
  }

  t_svd_produ_br(storage_type st) : st_(st) {}

  storage_type st_;
};

// By value return mechanism
template <typename T>
struct t_svd_produ_bv : Benchmark_base
{
  char const* what() { return "t_svd_produ_bv"; }
  float ops_per_point(length_type n)
  {
    float ops;
    // FLOP count for LAPACK DGESVD benchmark in the LAPACK Users Guide
    //  with matrix multiplication.
    if (st_ == svd_uvnos)
      ops = 4.67 * n * n * n - n * n;
    else
      ops = 8.67 * n * n * n - n * n;

    return impl::is_complex<T>::value ? (4.f * ops / n) : (ops / n);
  }

  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type size) { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef typename impl::scalar_of<T>::type scalar_type;

    Matrix<T>           A(size, size, T());
    Matrix<T>           C(size, size, T());
    Matrix<T>           X(size, size);
    Vector<scalar_type> V(size);

    randm(A);
    randm(C);

    svd<T, by_value> sv(size, size, st_, st_);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type i = 0; i < loop; ++i)
    {
      V = sv.decompose(A);
      X = sv.template produ<mat_ntrans, mat_lside>(C);
    }
    t1.stop();
    
    time = t1.delta();
  }

  t_svd_produ_bv(storage_type st) : st_(st) {}

  storage_type st_;
};

void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;
  loop.stop_ = 12;
}

int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_svd_br<float>(svd_uvnos)); break;
  case  2: loop(t_svd_br<float>(svd_uvpart)); break;
  case  3: loop(t_svd_br<float>(svd_uvfull)); break;
  case  4: loop(t_svd_br<std::complex<float> >(svd_uvnos)); break;
  case  5: loop(t_svd_br<std::complex<float> >(svd_uvpart)); break;
  case  6: loop(t_svd_br<std::complex<float> >(svd_uvfull)); break;

  case  11: loop(t_svd_bv<float>(svd_uvnos)); break;
  case  12: loop(t_svd_bv<float>(svd_uvpart)); break;
  case  13: loop(t_svd_bv<float>(svd_uvfull)); break;
  case  14: loop(t_svd_bv<std::complex<float> >(svd_uvnos)); break;
  case  15: loop(t_svd_bv<std::complex<float> >(svd_uvpart)); break;
  case  16: loop(t_svd_bv<std::complex<float> >(svd_uvfull)); break;

  case  22: loop(t_svd_produ_br<float>(svd_uvpart)); break;
  case  23: loop(t_svd_produ_br<float>(svd_uvfull)); break;
  case  25: loop(t_svd_produ_br<std::complex<float> >(svd_uvpart)); break;
  case  26: loop(t_svd_produ_br<std::complex<float> >(svd_uvfull)); break;

  case  32: loop(t_svd_produ_bv<float>(svd_uvpart)); break;
  case  33: loop(t_svd_produ_bv<float>(svd_uvfull)); break;
  case  35: loop(t_svd_produ_bv<std::complex<float> >(svd_uvpart)); break;
  case  36: loop(t_svd_produ_bv<std::complex<float> >(svd_uvfull)); break;

  case 0:
    std::cout
      << "svd -- Singular Value Decomposition.\n"
      << " Decompose :\n"
      << "   -1: by reference, float,   storage=svd_uvnos\n"
      << "   -2: by reference, float,   storage=svd_uvpart\n"
      << "   -3: by reference, float,   storage=svd_uvfull\n"
      << "   -4: by reference, complex, storage=svd_uvnos\n"
      << "   -5: by reference, complex, storage=svd_uvpart\n"
      << "   -6: by reference, complex, storage=svd_uvfull\n"
      << "\n"
      << "  -11: by     value, float,   storage=svd_uvnos\n"
      << "  -12: by     value, float,   storage=svd_uvpart\n"
      << "  -13: by     value, float,   storage=svd_uvfull\n"
      << "  -14: by     value, complex, storage=svd_uvnos\n"
      << "  -15: by     value, complex, storage=svd_uvpart\n"
      << "  -16: by     value, complex, storage=svd_uvfull\n"
      << "\n"
      << " Decompose with post product:\n"
      << "  -22: by reference, float,   storage=svd_uvpart\n"
      << "  -23: by reference, float,   storage=svd_uvfull\n"
      << "  -25: by reference, complex, storage=svd_uvpart\n"
      << "  -26: by reference, complex, storage=svd_uvfull\n"
      << "\n"
      << "  -32: by     value, float,   storage=svd_uvpart\n"
      << "  -33: by     value, float,   storage=svd_uvfull\n"
      << "  -35: by     value, complex, storage=svd_uvpart\n"
      << "  -36: by     value, complex, storage=svd_uvfull\n"
      ;
  default: return 0;
  }
  return 1;
}

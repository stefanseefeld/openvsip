/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/qrd.cpp
    @author  Jules Bergmann
    @date    2005-07-11
    @brief   VSIPL++ Library: Benchmark for QR factorization.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/solvers.hpp>

#include <vsip/opt/profile.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;

#define HAVE_LAPACK 0



/// Return a random value between -0.5 and +0.5

template <typename T>
struct Random
{
  static T value() { return T(1.f * rand()/(RAND_MAX+1.0)) - T(0.5); }
};

/// Specialization for random complex value.

template <typename T>
struct Random<complex<T> >
{
  static complex<T> value() {
    return complex<T>(Random<T>::value(), Random<T>::value());
  }
};



/// Fill a matrix with random values.

template <typename T,
	  typename Block>
void
randm(Matrix<T, Block> m)
{
  for (index_type r=0; r<m.size(0); ++r)
    for (index_type c=0; c<m.size(1); ++c)
      m(r, c) = Random<T>::value();
}



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
    return impl::Is_complex<T>::value ? (16.f * n * n /* n */ / 3)
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
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
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
    float ops = impl::Is_complex<T>::value ? (8.f * n * n * (3*m - n) / 3)
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
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      qr.decompose(A);
    t1.stop();
    
    time = t1.delta();
  }

  t_qrd(length_type ratio, storage_type st) : ratio_(ratio), st_(st) {}

  length_type  ratio_;
  storage_type st_;

};



void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;
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
      ;
  default: return 0;
  }
  return 1;
}

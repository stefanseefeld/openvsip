/* Copyright (c) 2005, 2006, 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/fft.cpp
    @author  Jules Bergmann
    @date    2005-08-24
    @brief   VSIPL++ Library: Benchmark for FFT.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip/opt/diag/fft.hpp>

#include "benchmarks.hpp"

using namespace vsip;


float
fft_ops(length_type len)
{
  return 5.0 * std::log((double)len) / std::log(2.0);
}



/***********************************************************************
  Fft, out-of-place
***********************************************************************/

template <typename T,
	  int      no_times>
struct t_fft_op : Benchmark_base
{
  char const* what() { return "t_fft_op"; }
  float ops_per_point(length_type len)  { return fft_ops(len); }
  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A(size, T());
    Vector<T>   Z(size);

    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times, alg_time>
      fft_type;

    fft_type fft(Domain<1>(size), scale_ ? (1.f/size) : 1.f);

    A = T(1);
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      fft(A, Z);
    t1.stop();
    
    if (!equal(Z.get(0), T(scale_ ? 1 : size)))
    {
      std::cout << "t_fft_op: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  void diag()
  {
    length_type size = 1024;

    Vector<T>   A(size, T());
    Vector<T>   Z(size);

    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times, alg_time>
      fft_type;

    fft_type fft(Domain<1>(size), scale_ ? (1.f/size) : 1.f);

    diagnose_fft_call("fft_op", fft, A, Z);
  }

  t_fft_op(bool scale) : scale_(scale) {}

  // Member data
  bool scale_;
};



/***********************************************************************
  Fft, in-place
***********************************************************************/

template <typename T,
	  int      no_times>
struct t_fft_ip : Benchmark_base
{
  char const* what() { return "t_fft_ip"; }
  float ops_per_point(length_type len)  { return fft_ops(len); }
  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A(size, T(0));

    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times, alg_time>
      fft_type;

    fft_type fft(Domain<1>(size), scale_ ? (1.f/size) : 1.f);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      fft(A);
    t1.stop();
    
    if (!equal(A.get(0), T(0)))
    {
      std::cout << "t_fft_ip: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  t_fft_ip(bool scale) : scale_(scale) {}

  // Member data
  bool scale_;
};



/***********************************************************************
  Fft, by-value
***********************************************************************/

template <typename T,
	  int      no_times>
struct t_fft_bv : Benchmark_base
{
  char const* what() { return "t_fft_bv"; }
  float ops_per_point (length_type len) { return fft_ops(len); }
  int riob_per_point(length_type)     { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type)     { return -1*(int)sizeof(T); }
  int mem_per_point (length_type)     { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A(size, T());
    Vector<T>   Z(size);

    typedef Fft<const_Vector, T, T, fft_fwd, by_value, no_times, alg_time>
      fft_type;

    fft_type fft(Domain<1>(size), scale_ ? (1.f/size) : 1.f);

    A = T(1);
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      Z = fft(A);
    t1.stop();
    
    test_assert(equal(Z.get(0), T(scale_ ? 1 : size)));
    
    time = t1.delta();
  }

  t_fft_bv(bool scale) : scale_(scale) {}

  // Member data
  bool scale_;
};



void
defaults(Loop1P& loop)
{
  loop.start_ = 4;
}



int
test(Loop1P& loop, int what)
{
  int const estimate = 1;  // FFT_ESTIMATE
  int const measure  = 15; // FFT_MEASURE (no_times > 12)
  int const patient  = 0;  // FFTW_PATIENT

  switch (what)
  {
#if VSIP_IMPL_PROVIDE_FFT_FLOAT
  case  1: loop(t_fft_op<complex<float>, estimate>(false)); break;
  case  2: loop(t_fft_ip<complex<float>, estimate>(false)); break;
  case  3: loop(t_fft_bv<complex<float>, estimate>(false)); break;
  case  5: loop(t_fft_op<complex<float>, estimate>(true)); break;
  case  6: loop(t_fft_ip<complex<float>, estimate>(true)); break;
  case  7: loop(t_fft_bv<complex<float>, estimate>(true)); break;

  case 11: loop(t_fft_op<complex<float>, measure>(false)); break;
  case 12: loop(t_fft_ip<complex<float>, measure>(false)); break;
  case 13: loop(t_fft_bv<complex<float>, measure>(false)); break;
  case 15: loop(t_fft_op<complex<float>, measure>(true)); break;
  case 16: loop(t_fft_ip<complex<float>, measure>(true)); break;
  case 17: loop(t_fft_bv<complex<float>, measure>(true)); break;

  case 21: loop(t_fft_op<complex<float>, patient>(false)); break;
  case 22: loop(t_fft_ip<complex<float>, patient>(false)); break;
  case 23: loop(t_fft_bv<complex<float>, patient>(false)); break;
  case 25: loop(t_fft_op<complex<float>, patient>(true)); break;
  case 26: loop(t_fft_ip<complex<float>, patient>(true)); break;
  case 27: loop(t_fft_bv<complex<float>, patient>(true)); break;
#endif

  // Double precision cases.

#if VSIP_IMPL_PROVIDE_FFT_DOUBLE
  case 101: loop(t_fft_op<complex<double>, estimate>(false)); break;
  case 102: loop(t_fft_ip<complex<double>, estimate>(false)); break;
  case 103: loop(t_fft_bv<complex<double>, estimate>(false)); break;
  case 105: loop(t_fft_op<complex<double>, estimate>(true)); break;
  case 106: loop(t_fft_ip<complex<double>, estimate>(true)); break;
  case 107: loop(t_fft_bv<complex<double>, estimate>(true)); break;

  case 111: loop(t_fft_op<complex<double>, measure>(false)); break;
  case 112: loop(t_fft_ip<complex<double>, measure>(false)); break;
  case 113: loop(t_fft_bv<complex<double>, measure>(false)); break;
  case 115: loop(t_fft_op<complex<double>, measure>(true)); break;
  case 116: loop(t_fft_ip<complex<double>, measure>(true)); break;
  case 117: loop(t_fft_bv<complex<double>, measure>(true)); break;

  case 121: loop(t_fft_op<complex<double>, patient>(false)); break;
  case 122: loop(t_fft_ip<complex<double>, patient>(false)); break;
  case 123: loop(t_fft_bv<complex<double>, patient>(false)); break;
  case 125: loop(t_fft_op<complex<double>, patient>(true)); break;
  case 126: loop(t_fft_ip<complex<double>, patient>(true)); break;
  case 127: loop(t_fft_bv<complex<double>, patient>(true)); break;
#endif

  case 0:
    std::cout
      << "fft -- Fft (fast fourier transform)\n"
#if VSIP_IMPL_PROVIDE_FFT_FLOAT
      << "Single precision\n"
      << " Planning effor: estimate (number of times = 1):\n"
      << "   -1 -- op: out-of-place CC fwd fft\n"
      << "   -2 -- ip: in-place     CC fwd fft\n"
      << "   -3 -- bv: by-value     CC fwd fft\n"
      << "   -5 -- op: out-of-place CC inv fft (w/scaling)\n"
      << "   -6 -- ip: in-place     CC inv fft (w/scaling)\n"
      << "   -7 -- bv: by-value     CC inv fft (w/scaling)\n"

      << " Planning effor: measure (number of times = 15): 11-16\n"
      << " Planning effor: pateint (number of times = 0): 21-26\n"
#else
      << "Single precision FFT support not provided by library\n"
#endif

      << "\n"
#if VSIP_IMPL_PROVIDE_FFT_DOUBLE
      << "\nDouble precision\n"
      << " Planning effor: estimate (number of times = 1): 101-106\n"
      << " Planning effor: measure (number of times = 15): 111-116\n"
      << " Planning effor: pateint (number of times = 0): 121-126\n"
#else
      << "Double precision FFT support not provided by library\n"
#endif
      ;

  default: return 0;
  }
  return 1;
}

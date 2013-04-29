/* Copyright (c) 2005, 2006, 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for IPP's FFT.

#include <iostream>
#include <cmath>
#include <complex>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/profile.hpp>

#include <ipps.h>

#include "benchmarks.hpp"

using std::complex;

bool check = true;

template <typename T> struct t_fft_inter_op;
template <typename T> struct t_fft_inter_ip;
template <typename T> struct t_fft_split_op;
template <typename T> struct t_fft_split_ip;



int
fft_ops(size_t len)
{
  return int(5 * std::log(len) / std::log(2));
}



template <>
struct t_fft_inter_op<complex<float> > : Benchmark_base
{
  char const* what() { return "t_fft_ipp"; }
  int ops_per_point(size_t len)  { return fft_ops(len); }
  int riob_per_point(size_t) { return -1*(int)sizeof(Ipp32fc); }
  int wiob_per_point(size_t) { return -1*(int)sizeof(Ipp32fc); }
  int mem_per_point(size_t)  { return  2*sizeof(Ipp32fc); }

  void operator()(size_t size, size_t loop, float& time)
  {
    Ipp32fc*            A;
    Ipp32fc*            Z;

    IppsFFTSpec_C_32fc* fft;
    Ipp8u*              buffer;

    int                 bufsize;
    int                 order = 0;

    for (size_t i=1; i<size; i <<= 1)
      order++;


    ippsFFTInitAlloc_C_32fc(&fft, order,
			    IPP_FFT_DIV_INV_BY_N, 
			    ippAlgHintFast);
    ippsFFTGetBufSize_C_32fc(fft, &bufsize);

    buffer = ippsMalloc_8u(bufsize);
    A      = ippsMalloc_32fc(size);
    Z      = ippsMalloc_32fc(size);

    if (!buffer || !A || !Z)
      throw(std::bad_alloc());
      

    for (size_t i=0; i<size; ++i)
    {
      A[i].re = 1.f;
      A[i].im = 0.f;
    }
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
    {
      ippsFFTFwd_CToC_32fc(A, Z, fft, buffer);
    }
    t1.stop();
    
    if (check)
    {
      test_assert(equal(Z[0].re, float(size)));
      test_assert(equal(Z[0].im, float(0)));
    }
    
    time = t1.delta();

    ippsFFTFree_C_32fc(fft);

    ippsFree(buffer);
    ippsFree(A);
    ippsFree(Z);
  }
};


#if IPP_VERSION_MAJOR >= 5
template <>
struct t_fft_inter_ip<complex<float> > : Benchmark_base
{
  char const* what() { return "t_fft_inter_ip"; }
  int ops_per_point(size_t len)  { return fft_ops(len); }
  int riob_per_point(size_t) { return -1*(int)sizeof(Ipp32fc); }
  int wiob_per_point(size_t) { return -1*(int)sizeof(Ipp32fc); }
  int mem_per_point(size_t)  { return  2*sizeof(Ipp32fc); }

  void operator()(size_t size, size_t loop, float& time)
  {
    Ipp32fc*            A;

    IppsFFTSpec_C_32fc* fft;
    Ipp8u*              buffer;

    int                 bufsize;
    int                 order = 0;

    for (size_t i=1; i<size; i <<= 1)
      order++;


    ippsFFTInitAlloc_C_32fc(&fft, order,
			    IPP_FFT_DIV_INV_BY_N, 
			    ippAlgHintFast);
    ippsFFTGetBufSize_C_32fc(fft, &bufsize);

    buffer = ippsMalloc_8u(bufsize);
    A      = ippsMalloc_32fc(size);

    if (!buffer || !A)
      throw(std::bad_alloc());
      

    for (size_t i=0; i<size; ++i)
    {
      A[i].re = 0.f;
      A[i].im = 0.f;
    }
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
    {
      ippsFFTFwd_CToC_32fc_I(A, fft, buffer);
    }
    t1.stop();
    
    if (check)
    {
      for (size_t i=0; i<size; ++i)
      {
	A[i].re = 0.f;
	A[i].im = 0.f;
      }
      ippsFFTFwd_CToC_32fc_I(A, fft, buffer);
      test_assert(equal(A[0].re, float(size)));
      test_assert(equal(A[0].im, float(0)));
    }
    
    time = t1.delta();

    ippsFFTFree_C_32fc(fft);

    ippsFree(buffer);
    ippsFree(A);
  }
};
#endif // IPP_VERSION_MAJOR >= 5



template <>
struct t_fft_split_op<complex<float> > : Benchmark_base
{
  char const* what() { return "t_fft_split_op"; }
  int ops_per_point(size_t len)  { return fft_ops(len); }
  int riob_per_point(size_t) { return -1*(int)sizeof(Ipp32fc); }
  int wiob_per_point(size_t) { return -1*(int)sizeof(Ipp32fc); }
  int mem_per_point(size_t)  { return  2*sizeof(Ipp32fc); }

  void operator()(size_t size, size_t loop, float& time)
  {
    Ipp32f*            Ar;
    Ipp32f*            Ai;
    Ipp32f*            Zr;
    Ipp32f*            Zi;

    IppsFFTSpec_C_32f* fft;
    Ipp8u*             buffer;

    int                 bufsize;
    int                 order = 0;

    for (size_t i=1; i<size; i <<= 1)
      order++;


    ippsFFTInitAlloc_C_32f(&fft, order,
			   IPP_FFT_DIV_INV_BY_N, 
			   ippAlgHintFast);
    ippsFFTGetBufSize_C_32f(fft, &bufsize);

    buffer = ippsMalloc_8u(bufsize);
    Ar     = ippsMalloc_32f(size);
    Ai     = ippsMalloc_32f(size);
    Zr     = ippsMalloc_32f(size);
    Zi     = ippsMalloc_32f(size);

    if (!buffer || !Ar || !Ai || !Zr || !Zi)
      throw(std::bad_alloc());
      

    for (size_t i=0; i<size; ++i)
    {
      Ar[i] = 1.f;
      Ai[i] = 0.f;
    }
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
    {
      ippsFFTFwd_CToC_32f(Ar, Ai, Zr, Zi, fft, buffer);
    }
    t1.stop();
    
    if (check)
    {
      test_assert(equal(Zr[0], float(size)));
      test_assert(equal(Zi[0], float(0)));
    }
    
    time = t1.delta();

    ippsFFTFree_C_32f(fft);

    ippsFree(buffer);
    ippsFree(Ar);
    ippsFree(Ai);
    ippsFree(Zr);
    ippsFree(Zi);
  }
};



#if IPP_VERSION_MAJOR >= 5
template <>
struct t_fft_split_ip<complex<float> > : Benchmark_base
{
  char const* what() { return "t_fft_split_ip"; }
  int ops_per_point(size_t len)  { return fft_ops(len); }
  int riob_per_point(size_t) { return -1*(int)sizeof(Ipp32fc); }
  int wiob_per_point(size_t) { return -1*(int)sizeof(Ipp32fc); }
  int mem_per_point(size_t)  { return  2*sizeof(Ipp32fc); }

  void operator()(size_t size, size_t loop, float& time)
  {
    Ipp32f*            Ar;
    Ipp32f*            Ai;

    IppsFFTSpec_C_32f* fft;
    Ipp8u*             buffer;

    int                bufsize;
    int                order = 0;

    for (size_t i=1; i<size; i <<= 1)
      order++;


    ippsFFTInitAlloc_C_32f(&fft, order,
			   IPP_FFT_DIV_INV_BY_N, 
			   ippAlgHintFast);
    ippsFFTGetBufSize_C_32f(fft, &bufsize);

    buffer = ippsMalloc_8u(bufsize);
    Ar     = ippsMalloc_32f(size);
    Ai     = ippsMalloc_32f(size);

    if (!buffer || !Ar || !Ai)
      throw(std::bad_alloc());
      

    for (size_t i=0; i<size; ++i)
    {
      Ar[i] = 1.f;
      Ai[i] = 0.f;
    }
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
    {
      ippsFFTFwd_CToC_32f_I(Ar, Ai, fft, buffer);
    }
    t1.stop();
    
    if (check)
    {
      for (size_t i=0; i<size; ++i)
      {
	Ar[i] = 1.f;
	Ai[i] = 0.f;
      }
      ippsFFTFwd_CToC_32f_I(Ar, Ai, fft, buffer);
      test_assert(equal(Ar[0], float(size)));
      test_assert(equal(Ai[0], float(0)));
    }
    
    time = t1.delta();

    ippsFFTFree_C_32f(fft);

    ippsFree(buffer);
    ippsFree(Ar);
    ippsFree(Ai);
  }
};
#endif // IPP_VERSION_MAJOR >= 5



void
defaults(Loop1P& loop)
{
  loop.start_ = 4;
}


int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_fft_inter_op<complex<float> >()); break;
#if IPP_VERSION_MAJOR >= 5
  case  2: loop(t_fft_inter_ip<complex<float> >()); break;
#endif // IPP_VERSION_MAJOR >= 5
  case 11: loop(t_fft_split_op<complex<float> >()); break;
#if IPP_VERSION_MAJOR >= 5
  case 12: loop(t_fft_split_ip<complex<float> >()); break;
#endif // IPP_VERSION_MAJOR >= 5

  case 0:
    std::cout
      << "ipp/fft -- IPP Fft (fast fourier transform) benchmark\n"
      << "Single precision\n"
      << "   -1 -- inter-op: interleaved, out-of-place CC fwd fft\n"
#if IPP_VERSION_MAJOR >= 5
      << "   -2 -- inter-ip: interleaved, in-place     CC fwd fft\n"
#endif // IPP_VERSION_MAJOR >= 5
      << "  -11 -- split-op: interleaved, out-of-place CC fwd fft\n"
#if IPP_VERSION_MAJOR >= 5
      << "  -12 -- split-ip: interleaved, out-of-place CC fwd fft\n"
#endif // IPP_VERSION_MAJOR >= 5
      ;

  default: return 0;
  }
  return 1;
}

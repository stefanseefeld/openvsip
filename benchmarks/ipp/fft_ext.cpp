/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for FFT using dda::Data to call IPP.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/profile.hpp>

#include <ipps.h>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;
using vsip_csl::equal;

int
get_order(length_type size)
{
  int order = 0;
  for (size_t i=1; i<size; i <<= 1)
    order++;
  return order;
}



template <typename T>
struct IppFFT;

template <>
struct IppFFT<std::complex<float> >
{
  typedef IppsFFTSpec_C_32fc* fft_type;
  typedef std::complex<float> in_type;
  typedef std::complex<float> out_type;

  static fft_type alloc(int order, int scale_flag)
  {
    fft_type fft;
    IppStatus status =
      ippsFFTInitAlloc_C_32fc(&fft, order, scale_flag, ippAlgHintFast);
    return (status == ippStsNoErr) ? fft : NULL;
  }

  static void free(fft_type fft)
  {
    ippsFFTFree_C_32fc(fft);
  }

  static int bufsize(fft_type fft)
  {
    int size;
    ippsFFTGetBufSize_C_32fc(fft, &size);
    return size;
  }

  static void forward(fft_type fft, in_type* in, out_type* out, Ipp8u* buffer)
  {
    ippsFFTFwd_CToC_32fc(reinterpret_cast<Ipp32fc*>(in),
			 reinterpret_cast<Ipp32fc*>(out), fft, buffer);
  }

  static void inverse(fft_type fft, in_type* in, out_type* out, Ipp8u* buffer)
  {
    ippsFFTInv_CToC_32fc(reinterpret_cast<Ipp32fc*>(in),
			 reinterpret_cast<Ipp32fc*>(out), fft, buffer);
  }
};

template <int      Dir,
	  typename T>
class ErsatzFFT
{
  // Constructor
public:
  ErsatzFFT(Domain<1> const& dom, float scale)
  {
    int scale_flag;

    if (scale == 1.f)
      scale_flag = IPP_FFT_NODIV_BY_ANY;
    else if (scale == dom.size())
    {
      if (Dir == -1)
	scale_flag = IPP_FFT_DIV_FWD_BY_N;
      else
	scale_flag = IPP_FFT_DIV_INV_BY_N;
    }


    fft_  = IppFFT<T>::alloc(get_order(dom.size()), scale_flag);
    if (fft_ == NULL)
      throw(std::bad_alloc());

    buffer_ = ippsMalloc_8u(IppFFT<T>::bufsize(fft_));
    if (buffer_ == NULL)
    {
      IppFFT<T>::free(fft_);
      throw(std::bad_alloc());
    }
  }

  template <typename Block1,
	    typename Block2>
  void
  operator()(const_Vector<T, Block1> in, Vector<T, Block2> out)
  {
    typedef Layout<1, row1_type, unit_stride, interleaved_complex> layout_t;

    dda::Data<Block1, dda::in, layout_t> in_data(in.block());
    dda::Data<Block2, dda::out, layout_t> out_data(out.block());

    assert(in_data.stride(0) == 1);
    assert(out_data.stride(0) == 1);

    if (Dir == -1)
      IppFFT<T>::forward(fft_, in_data.non_const_ptr(), out_data.ptr(), buffer_);
    else
      IppFFT<T>::inverse(fft_, in_data.non_const_ptr(), out_data.ptr(), buffer_);
  }


  // Member data.
private:
  typename IppFFT<T>::fft_type fft_;
  Ipp8u*                       buffer_;
};


int
fft_ops(length_type len)
{
  return int(5 * std::log(len) / std::log(2));
}



// Out of place FFT

template <typename T>
struct t_fft : Benchmark_base
{
  char const *what() { return "t_fft"; }
  int ops_per_point(length_type len)  { return fft_ops(len); }
  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type)  { return  2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A(size, T());
    Vector<T>   Z(size);

#if 0
    // int const no_times = 0; // FFTW_PATIENT
    int const no_times = 15; // not > 12 = FFT_MEASURE

    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times, alg_time>
      fft_type;
#else
    typedef ErsatzFFT<-1, T> fft_type;
#endif

    fft_type fft(Domain<1>(size), 1.f);

    A = T(1);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      fft(A, Z);
    t1.stop();
    
    if (!equal(Z(0), T(size)))
    {
      std::cout << "t_fft: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};



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
  case  1: loop(t_fft<complex<float> >()); break;
  case  0:
    std::cout
      << "fft_ext -- FFT using dda::Data to call IPP \n"
      << "\n"
      << "  -1: FFT on vector complex<float>\n"
      ;
  default: return 0;
  }
  return 1;
}

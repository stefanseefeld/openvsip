//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for SAL FFT.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>

#include "benchmarks.hpp"

using namespace vsip;

// Wrapper class for SAL FFTs.

template <typename T, storage_format_type C> struct sal_fft;

// SAL FFT for interleaved complex (INCOMPLETE).

template <>
struct sal_fft<float, interleaved_complex>
{
  typedef COMPLEX ctype;
};



// SAL FFT for split complex.

template <>
struct sal_fft<complex<float>, split_complex>
{
  typedef COMPLEX_SPLIT type;

  static type to_ptr(std::pair<float*, float*> const& ptr)
  {
    type ret = { ptr.first, ptr.second };
    return ret;
  }

  static type to_ptr(std::pair<float const*, float const*> const& ptr)
  {
    type ret = { (float*)ptr.first, (float*)ptr.second };
    return ret;
  }

  static void fftop(
    FFT_setup& setup,
    type in,
    type out,
    type tmp,
    int  size,
    int  dir)
  {
    long eflag = 0;  // no caching hints
    fft_zoptx(&setup, &in, 1, &out, 1, &tmp, size, dir, eflag);
  }

  static void fftip(
    FFT_setup& setup,
    type inout,
    type tmp,
    int  size,
    int  dir)
  {
    long eflag = 0;  // no caching hints
    fft_ziptx(&setup, &inout, 1, &tmp, size, dir, eflag);
  }

  static void scale(
    type        data,
    length_type size,
    float       s)
  {
    vsmulx(data.realp, 1, &s, data.realp, 1, size, 0);
    vsmulx(data.imagp, 1, &s, data.imagp, 1, size, 0);
  }

};



template <>
struct sal_fft<complex<float>, interleaved_complex>
{
  typedef COMPLEX* type;

  static COMPLEX*to_ptr(std::complex<float>* ptr) { return (type)ptr;}
  static COMPLEX*to_ptr(std::complex<float> const* ptr) { return (type)ptr;}

  static void fftop(
    FFT_setup& setup,
    type in,
    type out,
    type tmp,
    int  size,
    int  dir)
  {
    long eflag = 0;  // no caching hints
    fft_coptx(&setup, in, 2, out, 2, tmp, size, dir, eflag);
  }

  static void fftip(
    FFT_setup& setup,
    type inout,
    type tmp,
    int  size,
    int  dir)
  {
    long eflag = 0;  // no caching hints
    fft_ciptx(&setup, inout, 2, tmp, size, dir, eflag);
  }

  static void scale(
    type        data,
    length_type size,
    float       s)
  {
    float *d = reinterpret_cast<float*>(data);
    vsmulx(d, 1, &s, d, 1, 2 * size, 0);
  }

};



inline unsigned long
ilog2(length_type size)    // assume size = 2^n, != 0, return n.
{
  unsigned int n = 0;
  while (size >>= 1) ++n;
  return n;
}


int
fft_ops(length_type len)
{
  return int(5 * std::log((float)len) / std::log(2.f));
}


template <typename T, storage_format_type C>
struct t_fft_op : Benchmark_base
{
  typedef typename impl::scalar_of<T>::type scalar_type;

  char const *what() { return "t_fft_op"; }
  int ops_per_point(length_type len)  { return fft_ops(len); }
  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef sal_fft<T, C> traits;

    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A  (size, T());
    Vector<T, block_type>   tmp(size, T());
    Vector<T, block_type>   Z  (size);

    unsigned long log2N = ilog2(size);

    FFT_setup     setup;
    unsigned long nbytes  = 0;
    long          options = 0;
    long          dir     = FFT_FORWARD;
    scalar_type   factor  = scalar_type(1) / size;

    fft_setup(log2N, options, &setup, &nbytes);

    A = T(1);
    
    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_A(A.block());
      dda::Data<block_type, dda::in> ext_tmp(tmp.block());
      dda::Data<block_type, dda::out> ext_Z(Z.block());

      typename traits::type A_ptr = traits::to_ptr(ext_A.ptr());
      typename traits::type tmp_ptr = traits::to_ptr(ext_tmp.ptr());
      typename traits::type Z_ptr = traits::to_ptr(ext_Z.ptr());
    
      if (!scale_)
      {
	t1.start();
	for (index_type l=0; l<loop; ++l)
	  traits::fftop(setup, A_ptr, Z_ptr, tmp_ptr, log2N, dir);
	t1.stop();
      }
      else
      {
	t1.start();
	for (index_type l=0; l<loop; ++l)
	{
	  traits::fftop(setup, A_ptr, Z_ptr, tmp_ptr, log2N, dir);
	  traits::scale(Z_ptr, size, factor); 
	}
	t1.stop();
      }
    }
    
    if (!equal(Z.get(0), T(scale_ ? 1 : size)))
    {
      std::cout << "t_fft_op: ERROR" << std::endl;
      std::cout << "  got     : " << Z.get(0) << std::endl;
      std::cout << "  expected: " << T(scale_ ? 1 : size) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  t_fft_op(bool scale) : scale_(scale) {}

  // Member data
  bool scale_;
};



template <typename T, storage_format_type C>
struct t_fft_ip : Benchmark_base
{
  typedef typename impl::scalar_of<T>::type scalar_type;

  char const *what() { return "t_fft_ip"; }
  int ops_per_point(length_type len)  { return fft_ops(len); }
  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef sal_fft<T, C> traits;

    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A  (size, T());
    Vector<T, block_type>   tmp(size, T());

    unsigned long log2N = ilog2(size);

    FFT_setup     setup;
    unsigned long nbytes  = 0;
    long          options = 0;
    long          dir     = FFT_FORWARD;
    scalar_type   factor  = scalar_type(1) / size;

    fft_setup(log2N, options, &setup, &nbytes);

    A = T(0);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_A(A.block());
      dda::Data<block_type, dda::in> ext_tmp(tmp.block());

      typename traits::type A_ptr = traits::to_ptr(ext_A.ptr());
      typename traits::type tmp_ptr = traits::to_ptr(ext_tmp.ptr());
    
      if (!scale_)
      {
	t1.start();
	for (index_type l=0; l<loop; ++l)
	  traits::fftip(setup, A_ptr, tmp_ptr, log2N, dir);
	t1.stop();
	// Check answer
	A = T(1);
	traits::fftip(setup, A_ptr, tmp_ptr, log2N, dir);
      }
      else
      {
	t1.start();
	for (index_type l=0; l<loop; ++l)
	{
	  traits::fftip(setup, A_ptr, tmp_ptr, log2N, dir);
	  traits::scale(A_ptr, size, factor); 
	}
	t1.stop();
	// Check answer
	A = T(1);
	traits::fftip(setup, A_ptr, tmp_ptr, log2N, dir);
	traits::scale(A_ptr, size, factor); 
      }
    }
    
    if (!equal(A.get(0), T(scale_ ? 1 : size)))
    {
      std::cout << "t_fft_ip: ERROR" << std::endl;
      std::cout << "  got     : " << A.get(0) << std::endl;
      std::cout << "  expected: " << T(scale_ ? 1 : size) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  t_fft_ip(bool scale) : scale_(scale) {}

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
  switch (what)
  {
  case  1: loop(t_fft_op<complex<float>, split_complex>(false)); break;
  case  2: loop(t_fft_ip<complex<float>, split_complex>(false)); break;
  case  5: loop(t_fft_op<complex<float>, split_complex>(true)); break;
  case  6: loop(t_fft_ip<complex<float>, split_complex>(true)); break;

  case 11: loop(t_fft_op<complex<float>, interleaved_complex>(false)); break;
  case 12: loop(t_fft_ip<complex<float>, interleaved_complex>(false)); break;
  case 15: loop(t_fft_op<complex<float>, interleaved_complex>(true)); break;
  case 16: loop(t_fft_ip<complex<float>, interleaved_complex>(true)); break;

  case 0:
    std::cout
      << "fft -- SAL FFT \n"
      << "        in-place complex scale\n"
      << "   -1:      no    split    no  \n"
      << "   -2:     yes    split    no  \n"
      << "   -5:      no    split   yes \n"
      << "   -6:     yes    split   yes \n"
      << "\n"
      << "  -11:      no    inter    no  \n"
      << "  -12:     yes    inter    no  \n"
      << "  -15:      no    inter   yes \n"
      << "  -16:     yes    inter   yes\n"
      ;
  default: return 0;
  }
  return 1;
}

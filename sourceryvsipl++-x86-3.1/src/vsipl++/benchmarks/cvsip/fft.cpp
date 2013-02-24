/* Copyright (c) 2005, 2006, 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for VSIPL FFT.

#include "../benchmarks.hpp"
#include <vsip.h>
#include <iostream>

using namespace vsip;

float
fft_ops(length_type len)
{
  return 5.0 * std::log((double)len) / std::log(2.0);
}

template <typename T> struct fft_traits;

template <>
struct fft_traits<float>
{
  typedef vsip_vview_f input_type;
  typedef vsip_cvview_f output_type;
  typedef vsip_fft_f fft_type;

  static input_type *create_input(length_type l)
  {
    input_type *input = vsip_vcreate_f(l, VSIP_MEM_NONE);
    vsip_vfill_f(1.f, input);
    return input;
  }
  static void delete_input(input_type *v)
  { return vsip_valldestroy_f(v);}
  static output_type *create_output(length_type l)
  { return vsip_cvcreate_f(l, VSIP_MEM_NONE);}
  static void delete_output(output_type *v)
  { return vsip_cvalldestroy_f(v);}
  static fft_type *create_op_fft(length_type l, float s, int no_times)
  { return vsip_rcfftop_create_f(l, s, no_times, VSIP_ALG_TIME);}
  static void delete_fft(fft_type *f)
  { vsip_fft_destroy_f(f);}
  static void fft(fft_type *fft, input_type *input, output_type *output)
  { vsip_rcfftop_f(fft, input, output);}
  static bool valid(output_type *output, std::complex<float> const &value)
  {
    vsip_cscalar_f c = vsip_cvget_f(output, 0);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }


};

template <>
struct fft_traits<std::complex<float> >
{
  typedef vsip_cvview_f input_type;
  typedef vsip_cvview_f output_type;
  typedef vsip_fft_f fft_type;

  static input_type *create_input(length_type l)
  {
    input_type *input = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    vsip_cvfill_f(vsip_cmplx_f(1.f, 0.f), input);
    return input;
  }
  static void delete_input(input_type *v)
  { return vsip_cvalldestroy_f(v);}
  static output_type *create_output(length_type l)
  { return vsip_cvcreate_f(l, VSIP_MEM_NONE);}
  static void delete_output(output_type *v)
  { return vsip_cvalldestroy_f(v);}
  static fft_type *create_op_fft(length_type l, float s, int no_times)
  { return vsip_ccfftop_create_f(l, s, VSIP_FFT_FWD, no_times, VSIP_ALG_TIME);}
  static fft_type *create_ip_fft(length_type l, float s, int no_times)
  { return vsip_ccfftip_create_f(l, s, VSIP_FFT_FWD, no_times, VSIP_ALG_TIME);}
  static void delete_fft(fft_type *f)
  { vsip_fft_destroy_f(f);}
  static void fft(fft_type *fft, input_type *input, output_type *output)
  { vsip_ccfftop_f(fft, input, output);}
  static void fft(fft_type *fft, input_type *inout)
  { vsip_ccfftip_f(fft, inout);}
  static bool valid(output_type *output, std::complex<float> const &value)
  {
    vsip_cscalar_f c = vsip_cvget_f(output, 0);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }

};

template <>
struct fft_traits<double>
{
  typedef vsip_vview_d input_type;
  typedef vsip_cvview_d output_type;
  typedef vsip_fft_d fft_type;

  static input_type *create_input(length_type l)
  {
    input_type *input = vsip_vcreate_d(l, VSIP_MEM_NONE);
    vsip_vfill_d(1., input);
    return input;
  }
  static void delete_input(input_type *v)
  { return vsip_valldestroy_d(v);}
  static output_type *create_output(length_type l)
  { return vsip_cvcreate_d(l, VSIP_MEM_NONE);}
  static void delete_output(output_type *v)
  { return vsip_cvalldestroy_d(v);}
  static fft_type *create_op_fft(length_type l, float s, int no_times)
  { return vsip_rcfftop_create_d(l, s, no_times, VSIP_ALG_TIME);}
  static void delete_fft(fft_type *f)
  { vsip_fft_destroy_d(f);}
  static void fft(fft_type *fft, input_type *input, output_type *output)
  { vsip_rcfftop_d(fft, input, output);}
  static bool valid(output_type *output, std::complex<double> const &value)
  {
    vsip_cscalar_d c = vsip_cvget_d(output, 0);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }


};

template <>
struct fft_traits<std::complex<double> >
{
  typedef vsip_cvview_d input_type;
  typedef vsip_cvview_d output_type;
  typedef vsip_fft_d fft_type;

  static input_type *create_input(length_type l)
  {
    input_type *input = vsip_cvcreate_d(l, VSIP_MEM_NONE);
    vsip_cvfill_d(vsip_cmplx_d(1., 0.), input);
    return input;
  }
  static void delete_input(input_type *v)
  { return vsip_cvalldestroy_d(v);}
  static output_type *create_output(length_type l)
  { return vsip_cvcreate_d(l, VSIP_MEM_NONE);}
  static void delete_output(output_type *v)
  { return vsip_cvalldestroy_d(v);}
  static fft_type *create_op_fft(length_type l, float s, int no_times)
  { return vsip_ccfftop_create_d(l, s, VSIP_FFT_FWD, no_times, VSIP_ALG_TIME);}
  static fft_type *create_ip_fft(length_type l, float s, int no_times)
  { return vsip_ccfftip_create_d(l, s, VSIP_FFT_FWD, no_times, VSIP_ALG_TIME);}
  static void delete_fft(fft_type *f)
  { vsip_fft_destroy_d(f);}
  static void fft(fft_type *fft, input_type *input, output_type *output)
  { vsip_ccfftop_d(fft, input, output);}
  static void fft(fft_type *fft, input_type *inout)
  { vsip_ccfftip_d(fft, inout);}
  static bool valid(output_type *output, std::complex<double> const &value)
  {
    vsip_cscalar_d c = vsip_cvget_d(output, 0);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }

};


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
    typedef fft_traits<T> traits;
    typename traits::input_type *input = traits::create_input(size);
    typename traits::output_type *output = traits::create_output(size);
    typename traits::fft_type *fft = traits::create_op_fft(size,
                                                           scale_ ? 1.f/size : 1.f,
                                                           no_times);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      traits::fft(fft, input, output);
    }
    t1.stop();
    
    if (!traits::valid(output, T(scale_ ? 1 : size)))
      std::cout << "t_fft_op: ERROR" << std::endl;
    
    time = t1.delta();

    traits::delete_fft(fft);
    traits::delete_output(output);
    traits::delete_input(input);
  }

  t_fft_op(bool scale) : scale_(scale) {}

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
    typedef fft_traits<T> traits;
    typename traits::input_type *inout = traits::create_input(size);
    typename traits::fft_type *fft = traits::create_op_fft(size,
                                                           scale_ ? 1.f/size : 1.f,
                                                           no_times);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      traits::fft(fft, inout);
    }
    t1.stop();
    
    if (!traits::valid(inout, T(0)))
      std::cout << "t_fft_op: ERROR" << std::endl;
    
    time = t1.delta();

    traits::delete_fft(fft);
    traits::delete_input(inout);

  }

  t_fft_ip(bool scale) : scale_(scale) {}

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
  case  5: loop(t_fft_op<complex<float>, estimate>(true)); break;
  case  6: loop(t_fft_ip<complex<float>, estimate>(true)); break;

  case 11: loop(t_fft_op<complex<float>, measure>(false)); break;
  case 12: loop(t_fft_ip<complex<float>, measure>(false)); break;
  case 15: loop(t_fft_op<complex<float>, measure>(true)); break;
  case 16: loop(t_fft_ip<complex<float>, measure>(true)); break;

  case 21: loop(t_fft_op<complex<float>, patient>(false)); break;
  case 22: loop(t_fft_ip<complex<float>, patient>(false)); break;
  case 25: loop(t_fft_op<complex<float>, patient>(true)); break;
  case 26: loop(t_fft_ip<complex<float>, patient>(true)); break;
#endif

  // Double precision cases.

#if VSIP_IMPL_PROVIDE_FFT_DOUBLE
  case 101: loop(t_fft_op<complex<double>, estimate>(false)); break;
  case 102: loop(t_fft_ip<complex<double>, estimate>(false)); break;
  case 105: loop(t_fft_op<complex<double>, estimate>(true)); break;
  case 106: loop(t_fft_ip<complex<double>, estimate>(true)); break;

  case 111: loop(t_fft_op<complex<double>, measure>(false)); break;
  case 112: loop(t_fft_ip<complex<double>, measure>(false)); break;
  case 115: loop(t_fft_op<complex<double>, measure>(true)); break;
  case 116: loop(t_fft_ip<complex<double>, measure>(true)); break;

  case 121: loop(t_fft_op<complex<double>, patient>(false)); break;
  case 122: loop(t_fft_ip<complex<double>, patient>(false)); break;
  case 125: loop(t_fft_op<complex<double>, patient>(true)); break;
  case 126: loop(t_fft_ip<complex<double>, patient>(true)); break;
#endif

  case 0:
    std::cout
      << "fft -- Fft (fast fourier transform)\n"
#if VSIP_IMPL_PROVIDE_FFT_FLOAT
      << "Single precision\n"
      << " Planning effor: estimate (number of times = 1):\n"
      << "   -1 -- op: out-of-place CC fwd fft\n"
      << "   -2 -- ip: in-place     CC fwd fft\n"
      << "   -5 -- op: out-of-place CC inv fft (w/scaling)\n"
      << "   -6 -- ip: in-place     CC inv fft (w/scaling)\n"

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

//
// Copyright (c) 2005, 2006, 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include "../benchmarks.hpp"
#include <vsip.h>
#include <iostream>

using namespace vsip;
using namespace vsip_csl;

// Number of times.  Parameter to Fftm.
//  1 - 12       - FFTW3 backend uses FFTW_ESTIMATE
//  12 +         - FFTW3 backend uses FFTW_MEASURE
//  0 (infinite) - FFTW3 backend uses FFTW_PATIENT (slow for big sizes)
//
// Turns out that FFTW_ESTIMATE is nearly as good as MEASURE and PATIENT.
int const no_times = 1;



float
fft_ops(length_type len)
{
  return 5.0 * len * std::log((double)len) / std::log(2.0);
}

struct Impl_op;		// out-of-place
struct Impl_ip;		// in-place

template <typename T> struct fftm_traits;
template <>
struct fftm_traits<std::complex<float> >
{
  typedef vsip_cmview_f input_type;
  typedef vsip_cmview_f output_type;
  typedef vsip_fftm_f fftm_type;

  static input_type *create_input(length_type r, length_type c)
  {
    input_type *input = vsip_cmcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);
    vsip_cmfill_f(vsip_cmplx_f(1.f, 0.f), input);
    return input;
  }
  static void delete_input(input_type *v)
  { return vsip_cmalldestroy_f(v);}
  static output_type *create_output(length_type r, length_type c)
  { return vsip_cmcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);}
  static void delete_output(output_type *v)
  { return vsip_cmalldestroy_f(v);}
  static fftm_type *create_op_fftm(length_type r, length_type c, float s,
                                   vsip_major axis, int no_times)
  { return vsip_ccfftmop_create_f(r, c, s, VSIP_FFT_FWD, axis, no_times, VSIP_ALG_TIME);}
  static fftm_type *create_ip_fftm(length_type r, length_type c, float s,
                                   vsip_major axis, int no_times)
  { return vsip_ccfftmip_create_f(r, c, s, VSIP_FFT_FWD, axis, no_times, VSIP_ALG_TIME);}
  static void delete_fftm(fftm_type *f)
  { vsip_fftm_destroy_f(f);}
  static void fftm(fftm_type *fftm, input_type *input, output_type *output)
  { vsip_ccfftmop_f(fftm, input, output);}
  static void fftm(fftm_type *fftm, input_type *inout)
  { vsip_ccfftmip_f(fftm, inout);}
  static bool valid(output_type *output, std::complex<float> const &value)
  {
    vsip_cscalar_f c = vsip_cmget_f(output, 0, 0);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }

};

template <>
struct fftm_traits<std::complex<double> >
{
  typedef vsip_cmview_d input_type;
  typedef vsip_cmview_d output_type;
  typedef vsip_fftm_d fftm_type;

  static input_type *create_input(length_type r, length_type c)
  {
    input_type *input = vsip_cmcreate_d(r, c, VSIP_ROW, VSIP_MEM_NONE);
    vsip_cmfill_d(vsip_cmplx_d(1., 0.), input);
    return input;
  }
  static void delete_input(input_type *v)
  { return vsip_cmalldestroy_d(v);}
  static output_type *create_output(length_type r, length_type c)
  { return vsip_cmcreate_d(r, c, VSIP_ROW, VSIP_MEM_NONE);}
  static void delete_output(output_type *v)
  { return vsip_cmalldestroy_d(v);}
  static fftm_type *create_op_fftm(length_type r, length_type c, double s,
                                   vsip_major axis, int no_times)
  { return vsip_ccfftmop_create_d(r, c, s, VSIP_FFT_FWD, axis, no_times, VSIP_ALG_TIME);}
  static fftm_type *create_ip_fftm(length_type r, length_type c, double s,
                                   vsip_major axis, int no_times)
  { return vsip_ccfftmip_create_d(r, c, s, VSIP_FFT_FWD, axis, no_times, VSIP_ALG_TIME);}
  static void delete_fftm(fftm_type *f)
  { vsip_fftm_destroy_d(f);}
  static void fftm(fftm_type *fftm, input_type *input, output_type *output)
  { vsip_ccfftmop_d(fftm, input, output);}
  static void fftm(fftm_type *fftm, input_type *inout)
  { vsip_ccfftmip_d(fftm, inout);}
  static bool valid(output_type *output, std::complex<double> const &value)
  {
    vsip_cscalar_d c = vsip_cmget_d(output, 0, 0);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }

};

template <typename T,
	  typename Tag,
	  int      SD>
struct t_fftm;

/***********************************************************************
  Impl_op: Out-of-place Fftm
***********************************************************************/

template <typename T,
	  int      SD>
struct t_fftm<T, Impl_op, SD> : Benchmark_base
{
  static int const elem_per_point = 2;

  char const *what() { return "t_fftm<T, Impl_op, SD>"; }
  float ops(length_type rows, length_type cols)
  { return SD == row ? rows * fft_ops(cols) : cols * fft_ops(rows); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    typedef fftm_traits<T> traits;
    typename traits::input_type *input = traits::create_input(rows, cols);
    typename traits::output_type *output = traits::create_output(rows, cols);
    length_type size = SD == row ? cols : rows;
    typename traits::fftm_type *fftm = traits::create_op_fftm(rows, cols,
                                                              scale_ ? 1.f/size : 1.f,
                                                              SD == row ? VSIP_ROW : VSIP_COL,
                                                              no_times);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      traits::fftm(fftm, input, output);
    }
    t1.stop();
    
    if (!traits::valid(output, T(scale_ ? 1 : size)))
      std::cout << "t_fftm<T, Impl_op, SD>: ERROR" << std::endl;
    
    time = t1.delta();

    traits::delete_fftm(fftm);
    traits::delete_output(output);
    traits::delete_input(input);
  }

  t_fftm(bool scale) : scale_(scale) {}

  // Member data
  bool scale_;
};



/***********************************************************************
  Impl_ip: In-place Fftm
***********************************************************************/

template <typename T,
	  int      SD>
struct t_fftm<T, Impl_ip, SD> : Benchmark_base
{
  static int const elem_per_point = 1;

  char const *what() { return "t_fftm<T, Impl_ip, SD>"; }
  float ops(length_type rows, length_type cols)
    { return SD == row ? rows * fft_ops(cols) : cols * fft_ops(rows); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T>   A(rows, cols, T());

    typedef Fftm<T, T, SD, fft_fwd, by_reference, no_times, alg_time>
      fftm_type;

    fftm_type fftm(Domain<2>(rows, cols), 1.f);

    A = T(0);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      fftm(A);
    }
    t1.stop();

    A = T(1);
    fftm(A);
    
    if (!equal(A(0, 0), T(SD == row ? cols : rows)))
    {
      std::cout << "t_fftm<T, Impl_ip, SD>: ERROR" << std::endl;
      std::cout << "   got     : " << A(0, 0) << std::endl;
      std::cout << "   expected: " << T(SD == row ? cols : rows) << std::endl;
    }
    
    time = t1.delta();
  }

  t_fftm(bool scale) : scale_(scale) {}

  // Member data
  bool scale_;
};

/***********************************************************************
  Fixed rows driver
***********************************************************************/

template <typename T, typename ImplTag, int SD>
struct t_fftm_fix_rows : public t_fftm<T, ImplTag, SD>
{
  typedef t_fftm<T, ImplTag, SD> base_type;
  static int const elem_per_point = base_type::elem_per_point;

  char const *what() { return "t_fftm_fix_rows"; }
  float ops_per_point(length_type cols)
    { return (int)(this->ops(rows_, cols) / cols); }

  int riob_per_point(length_type) { return rows_*sizeof(T); }
  int wiob_per_point(length_type) { return rows_*sizeof(T); }
  int mem_per_point (length_type) { return rows_*elem_per_point*sizeof(T); }

  void operator()(length_type cols, length_type loop, float& time)
  {
    this->fftm(rows_, cols, loop, time);
  }

  t_fftm_fix_rows(length_type rows, bool scale)
    : base_type(scale), rows_(rows)
  {}

// Member data
  length_type rows_;
};



/***********************************************************************
  Fixed cols driver
***********************************************************************/

template <typename T, typename ImplTag, int SD>
struct t_fftm_fix_cols : public t_fftm<T, ImplTag, SD>
{
  typedef t_fftm<T, ImplTag, SD> base_type;
  static int const elem_per_point = base_type::elem_per_point;

  char const *what() { return "t_fftm_fix_cols"; }
  float ops_per_point(length_type rows)
    { return (int)(this->ops(rows, cols_) / rows); }
  int riob_per_point(length_type) { return cols_*sizeof(T); }
  int wiob_per_point(length_type) { return cols_*sizeof(T); }
  int mem_per_point (length_type) { return cols_*elem_per_point*sizeof(T); }

  void operator()(length_type rows, length_type loop, float& time)
  {
    this->fftm(rows, cols_, loop, time);
  }

  t_fftm_fix_cols(length_type cols, bool scale)
    : base_type(scale), cols_(cols)
  {}

// Member data
  length_type cols_;
};



/***********************************************************************
  Main definitions
***********************************************************************/

void
defaults(Loop1P& loop)
{
  loop.start_      = 4;
  loop.stop_       = 16;
  loop.loop_start_ = 10;

  loop.param_["rows"] = "64";
  loop.param_["size"] = "2048";
  loop.param_["scale"] = "0";
}



int
test(Loop1P& loop, int what)
{
  length_type rows  = atoi(loop.param_["rows"].c_str());
  length_type size  = atoi(loop.param_["size"].c_str());
  bool scale  = (loop.param_["scale"] == "1" ||
		 loop.param_["scale"] == "y");

  typedef complex<float>  Cf;
  typedef complex<double> Cd;

  switch (what)
  {
#if VSIP_IMPL_PROVIDE_FFT_FLOAT
  case  1: loop(t_fftm_fix_rows<Cf, Impl_op,   row>(rows, scale)); break;
  case  2: loop(t_fftm_fix_rows<Cf, Impl_ip,   row>(rows, false)); break;

  case 11: loop(t_fftm_fix_cols<Cf, Impl_op,   row>(size, false)); break;
  case 12: loop(t_fftm_fix_cols<Cf, Impl_ip,   row>(size, false)); break;

  case 21: loop(t_fftm_fix_rows<Cf, Impl_op,   row>(rows, true)); break;
#endif

#if VSIP_IMPL_PROVIDE_FFT_DOUBLE
  case 101: loop(t_fftm_fix_rows<Cd, Impl_op,   row>(rows, false)); break;
  case 102: loop(t_fftm_fix_rows<Cd, Impl_ip,   row>(rows, false)); break;

  case 111: loop(t_fftm_fix_cols<Cd, Impl_op,   row>(size, false)); break;
  case 112: loop(t_fftm_fix_cols<Cd, Impl_ip,   row>(size, false)); break;
#endif

  case 0:
    std::cout
      << "fftm -- Fftm (multiple fast fourier transform) benchmark\n"
      << "Single precision\n"
      << " Fixed rows, sweeping FFT size:\n"
      << "   -1 -- op  : out-of-place CC fwd fft\n"
      << "   -2 -- ip  : In-place CC fwd fft\n"
      << "\n"
      << " Parameters (for sweeping FFT size, cases 1 through 6)\n"
      << "  -p:rows ROWS -- set number of pulses (default 64)\n"
      << "\n"
      << " Fixed FFT size, sweeping number of FFTs:\n"
      << "  -11 -- op  : out-of-place CC fwd fft\n"
      << "  -12 -- ip  : In-place CC fwd fft\n"
      << "\n"
      << " Parameters (for sweeping number of FFTs, cases 11 through 16)\n"
      << "  -p:size SIZE -- size of pulse (default 2048)\n"
      ;

  default: return 0;
  }
  return 1;
}

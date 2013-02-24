/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/// Description
///   Benchmark for FFTW3 FFTM.

#include <iostream>
#include <fftw3.h>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>

#include "benchmarks.hpp"

using namespace vsip;

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
  return int(5 * len * std::log((float)len) / std::log(2.f));
}



template <typename T,
	  storage_format_type C,
	  typename Tag,
	  typename OrderT>
struct t_fftm;


// Implementation tags.

struct Impl_op;		// out-of-place



/***********************************************************************
  Impl_op: Out-of-place Fftm
***********************************************************************/

template <typename OrderT>
struct t_fftm<complex<float>, interleaved_complex, Impl_op, OrderT>
  : Benchmark_base
{
  static int const elem_per_point = 2;

  typedef complex<float>  T;
  static storage_format_type const ComplexFmt = interleaved_complex;

  char const* what() { return "t_fftm<..., Impl_op>"; }

  int ops(length_type rows, length_type cols)
    { return rows * fft_ops(cols); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    typedef Layout<2, OrderT, dense, ComplexFmt> LP;
    typedef impl::Strided<2, T, LP, Local_map> block_type;

    Matrix<T, block_type> A(rows, cols, T());
    Matrix<T, block_type> Z(rows, cols);
    
    vsip_csl::profile::Timer t1;
    {
      dda::Data<block_type, dda::in> ext_A(A.block());
      dda::Data<block_type, dda::out> ext_Z(Z.block());

      fftwf_plan p;

      int n[1], howmany;
      int istride, idist;
      int ostride, odist;

      n[0]    = cols;
      howmany = rows;
      istride = ext_A.stride(1);
      idist   = ext_A.stride(0);
      ostride = ext_Z.stride(1);
      odist   = ext_Z.stride(0);

      if (save_wisdom_)
	fftwf_forget_wisdom();

      p = fftwf_plan_many_dft(
		1,				// rank
		n,				// n - dim size
		howmany,			// homany
		(fftwf_complex*)(ext_A.ptr()), // in
		NULL,				// inembed
		istride,			// istride
		idist,				// idist
		(fftwf_complex*)(ext_Z.ptr()), // out
		NULL,				// onembed
		ostride,			// ostride
		odist,				// odist
		FFTW_FORWARD,			// sign
		flags_);			// flags

      // planning may scribble on data.
      A = T(1);
    
      t1.start();
      for (index_type l=0; l<loop; ++l)
	fftwf_execute(p);
      t1.stop();

      if (save_wisdom_)
      {
	char file[80];
	sprintf(file, "wisdom.%d-%d", (int)rows, (int)cols);
	FILE* fd = fopen(file, "w");
	fftwf_export_wisdom_to_file(fd);
	fclose(fd);
      }

      fftwf_destroy_plan(p);
    }
    
    if (!equal(Z.get(0, 0), T(scale_ ? 1.0 : cols)))
    {
      std::cout << "t_fft_op: ERROR" << std::endl;
      std::cout << "  got     : " << Z.get(0, 0) << std::endl;
      std::cout << "  expected: " << T(scale_ ? 1 : cols)
	        << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  t_fftm(bool scale, int flags, bool save_wisdom)
    : scale_      (scale),
      flags_      (flags),
      save_wisdom_(save_wisdom)
    {}

  // Member data
  bool scale_;
  int  flags_;
  bool save_wisdom_;
};



/***********************************************************************
  Fixed rows driver
***********************************************************************/

template <typename T, storage_format_type C, typename ImplTag, typename OrderT>
struct t_fftm_fix_rows : public t_fftm<T, C, ImplTag, OrderT>
{
  typedef t_fftm<T, C, ImplTag, OrderT> base_type;
  static int const elem_per_point = base_type::elem_per_point;

  char const* what() { return "t_fftm_fix_rows"; }
  int ops_per_point(length_type cols)
    { return (int)(this->ops(rows_, cols) / cols); }
  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type cols) { return cols*elem_per_point*sizeof(T); }

  void operator()(length_type cols, length_type loop, float& time)
  {
    this->fftm(rows_, cols, loop, time);
  }

  t_fftm_fix_rows(length_type rows, bool scale, int flags, bool sw)
    : base_type(scale, flags, sw), rows_(rows)
  {}

// Member data
  length_type rows_;
};



/***********************************************************************
  Fixed cols driver
***********************************************************************/

template <typename T, storage_format_type C, typename ImplTag, typename OrderT>
struct t_fftm_fix_cols : public t_fftm<T, C, ImplTag, OrderT>
{
  typedef t_fftm<T, C, ImplTag, OrderT> base_type;
  static int const elem_per_point = base_type::elem_per_point;

  char const* what() { return "t_fftm_fix_cols"; }
  int ops_per_point(length_type rows)
    { return (int)(this->ops(rows, cols_) / rows); }
  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type cols) { return cols*elem_per_point*sizeof(T); }

  void operator()(length_type rows, length_type loop, float& time)
  {
    this->fftm(rows, cols_, loop, time);
  }

  t_fftm_fix_cols(length_type cols, bool scale, int flags, bool sw)
    : base_type(scale, flags, sw), cols_(cols)
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
  loop.param_["sw"] = "0";
}



int
test(Loop1P& loop, int what)
{
  typedef complex<float>  C;
  static storage_format_type const I = interleaved_complex;
  typedef Impl_op         OP;
  typedef col2_type       C2;
  typedef row2_type       R2;

  length_type rows  = atoi(loop.param_["rows"].c_str());
  length_type size  = atoi(loop.param_["size"].c_str());
  // save wisdom
  bool sw  = (loop.param_["sw"] == "1" || loop.param_["sw"] == "y");

  int est = FFTW_ESTIMATE;
  int msr = FFTW_MEASURE;
  int pnt = FFTW_PATIENT;

  switch (what)
  {
  case  1: loop(t_fftm_fix_rows<C, I, OP, R2>(rows, false, est, sw)); break;
  case 11: loop(t_fftm_fix_cols<C, I, OP, R2>(size, false, est, sw)); break;
  case 21: loop(t_fftm_fix_rows<C, I, OP, C2>(rows, false, est, sw)); break;
  case 31: loop(t_fftm_fix_cols<C, I, OP, C2>(size, false, est, sw)); break;

  case 101: loop(t_fftm_fix_rows<C, I, OP, R2>(rows, false, msr, sw)); break;
  case 111: loop(t_fftm_fix_cols<C, I, OP, R2>(size, false, msr, sw)); break;
  case 121: loop(t_fftm_fix_rows<C, I, OP, C2>(rows, false, msr, sw)); break;
  case 131: loop(t_fftm_fix_cols<C, I, OP, C2>(size, false, msr, sw)); break;

  case 201: loop(t_fftm_fix_rows<C, I, OP, R2>(rows, false, pnt, sw)); break;
  case 211: loop(t_fftm_fix_cols<C, I, OP, R2>(size, false, pnt, sw)); break;
  case 221: loop(t_fftm_fix_rows<C, I, OP, C2>(rows, false, pnt, sw)); break;
  case 231: loop(t_fftm_fix_cols<C, I, OP, C2>(size, false, pnt, sw)); break;

  case 0:
    std::cout
      << "fftw3/fftm -- FFTW3 FFTM (multiple fast fourier transform) benchmark\n"
      << "Single precision, interlaved, ESTIMATE\n"
      << "   -1 -- out-of-place CC fwd fft, fixed rows, row-major\n"
      << "  -11 -- out-of-place CC fwd fft, fixed cols, row-major\n"
      << "  -21 -- out-of-place CC fwd fft, fixed rows, col-major\n"
      << "  -31 -- out-of-place CC fwd fft, fixed cols, col-major\n"
      << "\n"
      << "Single precision, interlaved, MEASURE\n"
      << " -1-1 -- out-of-place CC fwd fft, fixed rows, row-major\n"
      << " -111 -- out-of-place CC fwd fft, fixed cols, row-major\n"
      << " -121 -- out-of-place CC fwd fft, fixed rows, col-major\n"
      << " -131 -- out-of-place CC fwd fft, fixed cols, col-major\n"
      << "\n"
      << "Single precision, interlaved, PATIENT\n"
      << " -201 -- out-of-place CC fwd fft, fixed rows, row-major\n"
      << " -211 -- out-of-place CC fwd fft, fixed cols, row-major\n"
      << " -221 -- out-of-place CC fwd fft, fixed rows, col-major\n"
      << " -231 -- out-of-place CC fwd fft, fixed cols, col-major\n"
      << "\n"
      << " Parameters for all cases\n"
      << "  -p:sw [0|1] -- save wisdom (default 0)\n"
      << "\n"
      << " Parameters (for sweeping FFT size, cases 1 through 6)\n"
      << "  -p:rows ROWS -- set number of pulses (default 64)\n"
      << "\n"
      << " Parameters (for sweeping number of FFTs, cases 11 through 16)\n"
      << "  -p:size SIZE -- size of pulse (default 2048)\n"
      ;

  default: return 0;
  }
  return 1;
}

/* Copyright (c) 2005, 2006, 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/fftm.cpp
    @author  Jules Bergmann
    @date    2005-12-14
    @brief   VSIPL++ Library: Benchmark for Fftm.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip/opt/profile.hpp>
#include <vsip/opt/diag/fft.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;
using namespace vsip_csl;



/***********************************************************************
  Definitions
***********************************************************************/

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



template <typename T,
	  typename Tag,
	  int      SD>
struct t_fftm;

struct Impl_op;		// out-of-place
struct Impl_ip;		// in-place
struct Impl_pop;	// psuedo out-of-place (using OP FFT).
struct Impl_pip1;	// psuedo in-place (using in-place FFT).
struct Impl_pip2;	// psuedo in-place (using out-of-place FFT).
struct Impl_bv;		// by-value Fftm



/***********************************************************************
  Impl_op: Out-of-place Fftm
***********************************************************************/

template <typename T,
	  int      SD>
struct t_fftm<T, Impl_op, SD> : Benchmark_base
{
  static int const elem_per_point = 2;

  typedef Map<Block_dist, Whole_dist>      map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;
  typedef Matrix<T, block_type>            view_type;

  char const* what() { return "t_fftm<T, Impl_op, SD>"; }
  float ops(length_type rows, length_type cols)
    { return SD == row ? rows * fft_ops(cols) : cols * fft_ops(rows); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Whole_dist());

    view_type   A(rows, cols, T(), map);
    view_type   Z(rows, cols, map);

    typedef Fftm<T, T, SD, fft_fwd, by_reference, no_times, alg_time>
      fftm_type;

    length_type size = SD == row ? cols : rows;

    fftm_type fftm(Domain<2>(rows, cols), scale_ ? (1.f/size) : 1.f);

    A = T(1);
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      fftm(A, Z);
    t1.stop();
    
    if (!equal(Z(0, 0), T(scale_ ? 1.0 : SD == row ? cols : rows)))
    {
      std::cout << "t_fftm<T, Impl_op, SD>: ERROR" << std::endl;
      std::cout << "   got     : " << Z(0, 0) << std::endl;
      std::cout << "   expected: " << T(scale_ ? 1.0 : SD == row ? cols : rows)
		<< std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  void diag_rc(length_type rows, length_type cols)
  {
    Matrix<T>   A(rows, cols, T());
    Matrix<T>   Z(rows, cols);

    typedef Fftm<T, T, SD, fft_fwd, by_reference, no_times, alg_time>
      fftm_type;

    length_type size = SD == row ? cols : rows;

    fftm_type fftm(Domain<2>(rows, cols), scale_ ? (1.f/size) : 1.f);

    diagnose_fftm("fftm_op", fftm);
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

  typedef Map<Block_dist, Whole_dist>      map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;
  typedef Matrix<T, block_type>            view_type;

  char const* what() { return "t_fftm<T, Impl_ip, SD>"; }
  float ops(length_type rows, length_type cols)
    { return SD == row ? rows * fft_ops(cols) : cols * fft_ops(rows); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Whole_dist());

    view_type   A(rows, cols, T(), map);

    typedef Fftm<T, T, SD, fft_fwd, by_reference, no_times, alg_time>
      fftm_type;

    fftm_type fftm(Domain<2>(rows, cols), 1.f);

    A = T(0);
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      fftm(A);
    t1.stop();

    A = T(1);
    fftm(A);
    
    if (!equal(A(0, 0), T(SD == row ? cols : rows)))
    {
      std::cout << "t_fftm<T, Impl_ip, SD>: ERROR" << std::endl;
      std::cout << "   got     : " << A(0, 0) << std::endl;
      std::cout << "   expected: " << T(SD == row ? cols : rows) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  void diag_rc(length_type /*rows*/, length_type /*cols*/)
  {
    std::cout << "No diag\n";
  }

  t_fftm(bool scale) : scale_(scale) {}

  // Member data
  bool scale_;
};



/***********************************************************************
  Impl_pop: Pseudo out-of-place Fftm (using out-of-place Fft)
***********************************************************************/

template <typename T,
	  int      SD>
struct t_fftm<T, Impl_pop, SD> : Benchmark_base
{
  static int const elem_per_point = 1;

  char const* what() { return "t_fftm<T, Impl_pop, SD>"; }
  float ops(length_type rows, length_type cols)
    { return SD == row ? rows * fft_ops(cols) : cols * fft_ops(rows); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T>   A(rows, cols, T());
    Matrix<T>   Z(rows, cols);
    Vector<T>   tmp(SD == row ? cols : rows);

    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times, alg_time>
      fft_type;

    fft_type fft(Domain<1>(SD == row ? cols : rows), 1.f);

    A = T(1);
    
    vsip::impl::profile::Timer t1;
    
    if (SD == row)
    {
      t1.start();
      for (index_type l=0; l<loop; ++l)
	for (index_type i=0; i<rows; ++i)
	{
	  fft(A.row(i), Z.row(i));
	}
      t1.stop();
    }
    else
    {
      t1.start();
      for (index_type l=0; l<loop; ++l)
	for (index_type i=0; i<cols; ++i)
	{
	  fft(A.col(i), Z.col(i));
	}
      t1.stop();
    }

    if (!equal(Z(0, 0), T(SD == row ? cols : rows)))
    {
      std::cout << "t_fftm<T, Impl_pop, SD>: ERROR" << std::endl;
      std::cout << "   got     : " << Z(0, 0) << std::endl;
      std::cout << "   expected: " << T(SD == row ? cols : rows)
		<< std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  void diag_rc(length_type /*rows*/, length_type /*cols*/)
  {
    std::cout << "No diag\n";
  }

  t_fftm(bool scale) : scale_(scale) {}

  // Member data
  bool scale_;
};



/***********************************************************************
  Impl_pip1: Pseudo In-place Fftm (using in-place Fft)
***********************************************************************/

template <typename T,
	  int      SD>
struct t_fftm<T, Impl_pip1, SD> : Benchmark_base
{
  static int const elem_per_point = 1;

  char const* what() { return "t_fftm<T, Impl_pip1, SD>"; }
  float ops(length_type rows, length_type cols)
    { return SD == row ? rows * fft_ops(cols) : cols * fft_ops(rows); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T>   A(rows, cols, T());

    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times, alg_time>
      fft_type;

    fft_type fft(Domain<1>(SD == row ? cols : rows), 1.f);

    A = T(0);
    
    vsip::impl::profile::Timer t1;
    
    if (SD == row)
    {
      t1.start();
      for (index_type l=0; l<loop; ++l)
	for (index_type i=0; i<rows; ++i)
	  fft(A.row(i));
      t1.stop();

      A = T(1);
      for (index_type i=0; i<rows; ++i)
	fft(A.row(i));
    }
    else
    {
      t1.start();
      for (index_type l=0; l<loop; ++l)
	for (index_type i=0; i<cols; ++i)
	  fft(A.col(i));
      t1.stop();

      A = T(1);
      for (index_type i=0; i<cols; ++i)
	fft(A.col(i));
    }

    if (!equal(A(0, 0), T(SD == row ? cols : rows)))
    {
      std::cout << "t_fftm<T, Impl_pip1, SD>: ERROR" << std::endl;
      std::cout << "   got     : " << A(0, 0) << std::endl;
      std::cout << "   expected: " << T(SD == row ? cols : rows)
		<< std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  void diag_rc(length_type /*rows*/, length_type /*cols*/)
  {
    std::cout << "No diag\n";
  }

  t_fftm(bool scale) : scale_(scale) {}

  // Member data
  bool scale_;
};



/***********************************************************************
  Impl_pip2: Pseudo In-place Fftm (using out-of-place Fft)
***********************************************************************/

template <typename T,
	  int      SD>
struct t_fftm<T, Impl_pip2, SD> : Benchmark_base
{
  static int const elem_per_point = 1;

  char const* what() { return "t_fftm<T, Impl_pip2, SD>"; }
  float ops(length_type rows, length_type cols)
    { return SD == row ? rows * fft_ops(cols) : cols * fft_ops(rows); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T>   A(rows, cols, T());
    Vector<T>   tmp(SD == row ? cols : rows);

    typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times, alg_time>
      fft_type;

    fft_type fft(Domain<1>(SD == row ? cols : rows), 1.f);

    A = T(0);
    
    vsip::impl::profile::Timer t1;
    
    if (SD == row)
    {
      t1.start();
      for (index_type l=0; l<loop; ++l)
	for (index_type i=0; i<rows; ++i)
	{
	  fft(A.row(i), tmp);
	  A.row(i) = tmp;
	}
      t1.stop();

      A = T(1);
      for (index_type i=0; i<rows; ++i)
      {
	fft(A.row(i), tmp);
	A.row(i) = tmp;
      }
    }
    else
    {
      t1.start();
      for (index_type l=0; l<loop; ++l)
	for (index_type i=0; i<cols; ++i)
	{
	  fft(A.col(i), tmp);
	  A.col(i) = tmp;
	}
      t1.stop();

      A = T(1);
      for (index_type i=0; i<cols; ++i)
      {
	fft(A.col(i), tmp);
	A.col(i) = tmp;
      }
    }

    if (!equal(A(0, 0), T(SD == row ? cols : rows)))
    {
      std::cout << "t_fftm<T, Impl_pip2, SD>: ERROR" << std::endl;
      std::cout << "   got     : " << A(0, 0) << std::endl;
      std::cout << "   expected: " << T(SD == row ? cols : rows)
		<< std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  void diag_rc(length_type /*rows*/, length_type /*cols*/)
  {
    std::cout << "No diag\n";
  }

  t_fftm(bool scale) : scale_(scale) {}

  // Member data
  bool scale_;
};



/***********************************************************************
  Impl_bv: By-value Fftm
***********************************************************************/

template <typename T,
	  int      SD>
struct t_fftm<T, Impl_bv, SD> : Benchmark_base
{
  static int const elem_per_point = 2;

  char const* what() { return "t_fftm<T, Impl_bv, SD>"; }
  float ops(length_type rows, length_type cols)
    { return SD == row ? rows * fft_ops(cols) : cols * fft_ops(rows); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T>   A(rows, cols, T());
    Matrix<T>   Z(rows, cols);

    typedef Fftm<T, T, SD, fft_fwd, by_value, no_times, alg_time>
      fftm_type;

    length_type size = SD == row ? cols : rows;

    fftm_type fftm(Domain<2>(rows, cols), scale_ ? (1.f/size) : 1.f);

    A = T(1);
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      Z = fftm(A);
    t1.stop();
    
    test_assert(equal(Z.get(0, 0), T(scale_ ? 1.0 : SD == row ? cols : rows)));
    
    time = t1.delta();
  }

  void diag_rc(length_type /*rows*/, length_type /*cols*/)
  {
    std::cout << "No diag\n";
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

  char const* what() { return "t_fftm_fix_rows"; }
  float ops_per_point(length_type cols)
    { return (int)(this->ops(rows_, cols) / cols); }

  int riob_per_point(length_type) { return rows_*sizeof(T); }
  int wiob_per_point(length_type) { return rows_*sizeof(T); }
  int mem_per_point (length_type) { return rows_*elem_per_point*sizeof(T); }

  void operator()(length_type cols, length_type loop, float& time)
  {
    this->fftm(rows_, cols, loop, time);
  }

  void diag()
  {
    this->diag_rc(rows_, (vsip::length_type)1024);
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

  char const* what() { return "t_fftm_fix_cols"; }
  float ops_per_point(length_type rows)
    { return (int)(this->ops(rows, cols_) / rows); }
  int riob_per_point(length_type) { return cols_*sizeof(T); }
  int wiob_per_point(length_type) { return cols_*sizeof(T); }
  int mem_per_point (length_type) { return cols_*elem_per_point*sizeof(T); }

  void operator()(length_type rows, length_type loop, float& time)
  {
    this->fftm(rows, cols_, loop, time);
  }

  void diag()
  {
    this->diag_rc((vsip::length_type)64, cols_);
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
  case  3: loop(t_fftm_fix_rows<Cf, Impl_pop,  row>(rows, false)); break;
  case  4: loop(t_fftm_fix_rows<Cf, Impl_pip1, row>(rows, false)); break;
  case  5: loop(t_fftm_fix_rows<Cf, Impl_pip2, row>(rows, false)); break;
  case  6: loop(t_fftm_fix_rows<Cf, Impl_bv,   row>(rows, false)); break;

  case 11: loop(t_fftm_fix_cols<Cf, Impl_op,   row>(size, false)); break;
  case 12: loop(t_fftm_fix_cols<Cf, Impl_ip,   row>(size, false)); break;
  case 13: loop(t_fftm_fix_cols<Cf, Impl_pop,  row>(size, false)); break;
  case 14: loop(t_fftm_fix_cols<Cf, Impl_pip1, row>(size, false)); break;
  case 15: loop(t_fftm_fix_cols<Cf, Impl_pip2, row>(size, false)); break;
  case 16: loop(t_fftm_fix_cols<Cf, Impl_bv,   row>(size, false)); break;

#if 0
  case 11: loop(t_fftm_fix_rows<complex<float>, Impl_op,   col>(p)); break;
  case 12: loop(t_fftm_fix_rows<complex<float>, Impl_ip,   col>(p)); break;
  case 13: loop(t_fftm_fix_rows<complex<float>, Impl_pip1, col>(p)); break;
  case 14: loop(t_fftm_fix_rows<complex<float>, Impl_pip2, col>(p)); break;
  case 15: loop(t_fftm_fix_cols<complex<float>, Impl_op,   col>(p)); break;
  case 16: loop(t_fftm_fix_cols<complex<float>, Impl_ip,   col>(p)); break;
  case 17: loop(t_fftm_fix_cols<complex<float>, Impl_pip1, col>(p)); break;
  case 18: loop(t_fftm_fix_cols<complex<float>, Impl_pip2, col>(p)); break;
#endif

  case 21: loop(t_fftm_fix_rows<Cf, Impl_op,   row>(rows, true)); break;
#endif

#if VSIP_IMPL_PROVIDE_FFT_DOUBLE
  case 101: loop(t_fftm_fix_rows<Cd, Impl_op,   row>(rows, false)); break;
  case 102: loop(t_fftm_fix_rows<Cd, Impl_ip,   row>(rows, false)); break;
  case 103: loop(t_fftm_fix_rows<Cd, Impl_pop,  row>(rows, false)); break;
  case 104: loop(t_fftm_fix_rows<Cd, Impl_pip1, row>(rows, false)); break;
  case 105: loop(t_fftm_fix_rows<Cd, Impl_pip2, row>(rows, false)); break;
  case 106: loop(t_fftm_fix_rows<Cd, Impl_bv,   row>(rows, false)); break;

  case 111: loop(t_fftm_fix_cols<Cd, Impl_op,   row>(size, false)); break;
  case 112: loop(t_fftm_fix_cols<Cd, Impl_ip,   row>(size, false)); break;
  case 113: loop(t_fftm_fix_cols<Cd, Impl_pop,  row>(size, false)); break;
  case 114: loop(t_fftm_fix_cols<Cd, Impl_pip1, row>(size, false)); break;
  case 115: loop(t_fftm_fix_cols<Cd, Impl_pip2, row>(size, false)); break;
  case 116: loop(t_fftm_fix_cols<Cd, Impl_bv,   row>(size, false)); break;
#endif

  case 0:
    std::cout
      << "fftm -- Fftm (multiple fast fourier transform) benchmark\n"
      << "Single precision\n"
      << " Fixed rows, sweeping FFT size:\n"
      << "   -1 -- op  : out-of-place CC fwd fft\n"
      << "   -2 -- ip  : In-place CC fwd fft\n"
      << "   -3 -- pop : Psuedo out-of-place CC fwd fft\n"
      << "   -4 -- pip1: Psuedo in-place v1 CC fwd fft\n"
      << "   -5 -- pip2: Psuedo in-place v2 CC fwd fft\n"
      << "   -6 -- bv  : By-value CC fwd fft\n"
      << "\n"
      << " Parameters (for sweeping FFT size, cases 1 through 6)\n"
      << "  -p:rows ROWS -- set number of pulses (default 64)\n"
      << "\n"
      << " Fixed FFT size, sweeping number of FFTs:\n"
      << "  -11 -- op  : out-of-place CC fwd fft\n"
      << "  -12 -- ip  : In-place CC fwd fft\n"
      << "  -13 -- pop : Psuedo out-of-place CC fwd fft\n"
      << "  -14 -- pip1: Psuedo in-place v1 CC fwd fft\n"
      << "  -15 -- pip2: Psuedo in-place v2 CC fwd fft\n"
      << "  -16 -- bv  : By-value CC fwd fft\n"
      << "\n"
      << " Parameters (for sweeping number of FFTs, cases 11 through 16)\n"
      << "  -p:size SIZE -- size of pulse (default 2048)\n"
      ;

  default: return 0;
  }
  return 1;
}

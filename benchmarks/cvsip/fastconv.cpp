/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for VSIPL Fast Convolution.

#include "../benchmarks.hpp"
#include "../fastconv.hpp"
#include <vsip.h>
#include <iostream>
#include <vsip_csl/output.hpp>

using namespace vsip;
using namespace vsip_csl;

#define DEBUG 0


struct Impl1op;		// out-of-place, phased fast-convolution


template <typename T> struct fftm_traits;
template <>
struct fftm_traits<std::complex<float> >
{
  typedef vsip_cmview_f matrix_type;
  typedef vsip_fftm_f fftm_type;

  static matrix_type *create_matrix(length_type r, length_type c, 
    std::complex<float> const& value)
  {
    matrix_type *view = vsip_cmcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);
    vsip_cmfill_f(vsip_cmplx_f(value.real(), value.imag()), view);
    return view;
  }
  static void delete_matrix(matrix_type *m)
  { return vsip_cmalldestroy_f(m);}

  static fftm_type *create_op_fftm(length_type r, length_type c, float s,
                                   vsip_major axis, int no_times)
  { return vsip_ccfftmop_create_f(r, c, s, VSIP_FFT_FWD, axis, no_times, VSIP_ALG_TIME);}
  static fftm_type *create_ip_fftm(length_type r, length_type c, float s,
                                   vsip_major axis, int no_times)
  { return vsip_ccfftmip_create_f(r, c, s, VSIP_FFT_FWD, axis, no_times, VSIP_ALG_TIME);}
  static fftm_type *create_op_inv_fftm(length_type r, length_type c, float s,
                                   vsip_major axis, int no_times)
  { return vsip_ccfftmop_create_f(r, c, s, VSIP_FFT_INV, axis, no_times, VSIP_ALG_TIME);}
  static fftm_type *create_ip_inv_fftm(length_type r, length_type c, float s,
                                   vsip_major axis, int no_times)
  { return vsip_ccfftmip_create_f(r, c, s, VSIP_FFT_INV, axis, no_times, VSIP_ALG_TIME);}
  static void delete_fftm(fftm_type *f)
  { vsip_fftm_destroy_f(f);}

  static void get(index_type i, index_type j, matrix_type *output, std::complex<float>& value)
  {
    vsip_cscalar_f cval;
    cval = vsip_cmget_f(output, i, j);
    value.real() = cval.r;
    value.imag() = cval.i;
  }
  static void put(index_type i, index_type j, matrix_type *input, std::complex<float> value)
  {
    vsip_cscalar_f cval;
    cval.r = value.real();
    cval.i = value.imag();
    vsip_cmput_f(input, i, j, cval);
  }

  static void fftm(fftm_type *fftm, matrix_type *input, matrix_type *output)
  { vsip_ccfftmop_f(fftm, input, output);}
  static void fftm(fftm_type *fftm, matrix_type *inout)
  { vsip_ccfftmip_f(fftm, inout);}
  static bool valid(matrix_type *output, std::complex<float> const &value)
  {
    vsip_cscalar_f c = vsip_cmget_f(output, 0, 0);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }

};

template <typename T1> struct vmmul_traits;
template <>
struct vmmul_traits<std::complex<float> >
{
  typedef vsip_cvview_f vector_type;
  typedef vsip_cmview_f matrix_type;

  static void initialize()
  { vsip_init((void *)0); }

  static vector_type *create_vector(length_type l, 
    std::complex<float> const& value)
  {
    vector_type *view = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    vsip_cvfill_f(vsip_cmplx_f(0.f, 0.f), view);
    return view;
  }
  static matrix_type *create_matrix(length_type r, length_type c, 
    std::complex<float> const& value)
  {
    matrix_type *view = vsip_cmcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);
    vsip_cmfill_f(vsip_cmplx_f(value.real(), value.imag()), view);
    return view;
  }
  static void delete_vector(vector_type *v) { vsip_cvalldestroy_f(v); }
  static void delete_matrix(matrix_type *m) { vsip_cmalldestroy_f(m); }

  static void get(index_type i, vector_type* output, std::complex<float>& value)
  {
    vsip_cscalar_f cval;
    cval = vsip_cvget_f(output, i);
    value.real() = cval.r;
    value.imag() = cval.i;
  }
  static void put(index_type i, vector_type* input, std::complex<float> value)
  {
    vsip_cscalar_f cval;
    cval.r = value.real();
    cval.i = value.imag();
    vsip_cvput_f(input, i, cval);
  }

  static void vmmul(vector_type const *v, matrix_type const *m, vsip_major sd, 
    matrix_type *r)
  { vsip_cvmmul_f(v, m, sd, r); }

  static bool equal(vector_type const *v, matrix_type const *m, vsip_major sd, 
    matrix_type const *r,
    index_type i, index_type j)
  {
    index_type idx = (sd == VSIP_ROW ? j : i);
    vsip_cscalar_f lhs = vsip_cmul_f(vsip_cvget_f(v, idx), vsip_cmget_f(m, i, j));
    vsip_cscalar_f rhs = vsip_cmget_f(r, i, j);
    return ::equal(lhs.r, rhs.r) && ::equal(lhs.i, rhs.i);
  }

  static void finalize()
  { vsip_finalize((void *)0); }
};




/***********************************************************************
  Impl1op: out-of-place, phased fast-convolution
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl1op> : fastconv_ops
{
  static length_type const num_args = 2;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef fftm_traits<T> ftraits;
    typedef vmmul_traits<T> vtraits;
    typename ftraits::fftm_type *fftm = ftraits::create_op_fftm(
      npulse, nrange, 1.f, VSIP_ROW, 1);
    typename ftraits::fftm_type *inv_fftm = ftraits::create_op_fftm(
      npulse, nrange, 1.f / nrange, VSIP_ROW, 1);

    typename vtraits::matrix_type *data = vtraits::create_matrix(npulse, nrange, T(1));
    typename vtraits::matrix_type *out = vtraits::create_matrix(npulse, nrange, T());
    typename vtraits::matrix_type *tmp = vtraits::create_matrix(npulse, nrange, T());
    typename vtraits::vector_type *replica = vtraits::create_vector(npulse, T());
    vtraits::put(0, replica, T(1));

    vsip_csl::profile::Timer t1;
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      ftraits::fftm(fftm, data, tmp);
      vtraits::vmmul(replica, tmp, VSIP_ROW, tmp);
      ftraits::fftm(fftm, tmp, out);
    }
    t1.stop();
    time = t1.delta();
    

#if DEBUG
    std::cout << npulse << " x " << nrange << std::endl;
    for (index_type j = 0; j < 10; ++j)
    {
      T value = 0;
      ftraits::get(0, j, out, value);
      std::cout << value << " ";
    }
    std::cout << std::endl;
#endif

    if (!ftraits::valid(out, T(nrange)))
      std::cout << "t_fastconv<T, Impl1op>: ERROR" << std::endl;
    
    ftraits::delete_fftm(inv_fftm);
    ftraits::delete_fftm(fftm);
    vtraits::delete_vector(replica);
    vtraits::delete_matrix(tmp);
    vtraits::delete_matrix(data);
  }
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

  std::cout << "rows: " << rows << "  size: " << size 
	    << std::endl;

  switch (what)
  {
  case  1: loop(t_fastconv_pf<complex<float>, Impl1op>(rows)); break;

  case 11: loop(t_fastconv_rf<complex<float>, Impl1op>(size)); break;

  case 0:
    std::cout
      << "fastconv -- fast convolution benchmark\n"
      << " Sweeping pulse size:\n"
      << "   -1 -- Out-of-place, phased\n"
      << "\n"
      << " Parameters (for sweeping convolution size, cases 1 through 10)\n"
      << "  -p:rows ROWS -- set number of pulses (default 64)\n"
      << "\n"
      << " Sweeping number of pulses:\n"
      << "  -11 -- Out-of-place, phased\n"
      << "\n"
      << " Parameters (for sweeping number of convolutions, cases 11 through 20)\n"
      << "  -p:size SIZE -- size of pulse (default 2048)\n"
      << "\n"
      << " Common Parameters\n"
      << "  -p:check {0,n}|{1,y} -- check results (default 'y')\n"
      ;

  default: return 0;
  }
  return 1;
}

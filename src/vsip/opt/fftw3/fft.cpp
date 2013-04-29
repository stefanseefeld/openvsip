/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/fftw3/fft.cpp
    @author  Stefan Seefeld
    @date    2006-04-10
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with 
             FFTW3.
*/

#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <fftw3.h>

// We need to include this create_plan.hpp header file because fft_impl.cpp
// uses this file. We cannot include this file in fft_impl.cpp because
// fft_impl.cpp gets included multiple times here.
#include <vsip/opt/fftw3/create_plan.hpp>

// 070729: FFTW 3.1.2's split in-place complex-to-complex FFT is
// broken on PowerPC and x86.  The plan captures the gap between the
// real and imaginary components.
//
// 070730: FFTW 3.1.2 split out-of-place complex FFT is broken on
// PowerPC.
//
// 070730: FFTW 3.1.2's split real-complex and complex-real FFT
// also appear broken on x86.
//
// Brave souls may set
//   USE_FFTW_SPLIT 1
//   USE_BROKEN_FFTW_SPLIT 0
// to attempt a work-around: copy, then perform transform out-of-place.

// Control whether FFTW split-complex transforms are performed at all.
#define USE_FFTW_SPLIT 0

// Control whether a subset broken FFTW split-complex transforms are
// worked around.
#define USE_BROKEN_FFTW_SPLIT 0

namespace vsip
{
namespace impl
{
namespace fftw3
{

#if USE_FFTW_SPLIT
storage_format_type const fftw3_storage_format = vsip::impl::dense_complex_format;
#else
storage_format_type const fftw3_storage_format = interleaved_complex;
#endif

inline int
convert_NoT(unsigned int number)
{
  // a number value of '0' means 'infinity', and so is captured
  // by a wrap-around.
  if (number - 1 > 30) return FFTW_PATIENT;
  if (number - 1 > 10) return FFTW_MEASURE;
  return FFTW_ESTIMATE;
}

template <dimension_type D, typename I, typename O> struct Fft_base;
template <dimension_type D, typename I, typename O, int S> class Fft_impl;
template <typename I, typename O, int A, int D> class Fftm_impl;

} // namespace vsip::impl::fftw3
} // namespace vsip::impl
} // namespace vsip

#ifdef VSIP_IMPL_FFTW3_HAVE_FLOAT
#  define FFTW(fun) fftwf_##fun
#  define SCALAR_TYPE float
#  include "fft_impl.cpp"
#  undef SCALAR_TYPE
#  undef FFTW
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_DOUBLE
#  define FFTW(fun) fftw_##fun
#  define SCALAR_TYPE double
#  include "fft_impl.cpp"
#  undef SCALAR_TYPE
#  undef FFTW
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_LONG_DOUBLE
#  define FFTW(fun) fftwl_##fun
#  define SCALAR_TYPE long double
#  include "fft_impl.cpp"
#  undef SCALAR_TYPE
#  undef FFTW
#endif

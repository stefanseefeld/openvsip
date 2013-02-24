/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ref_dft.cpp
    @author  Jules Bergmann
    @date    2006-03-03
    @brief   VSIPL++ CodeSourcery Library: Reference implementation of 
             Discrete Fourier Transform function.
*/

#ifndef VSIP_CSL_REF_DFT_HPP
#define VSIP_CSL_REF_DFT_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>
#include <cassert>

#include <vsip/complex.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>


namespace vsip_csl
{

/***********************************************************************
  Definitions
***********************************************************************/

namespace ref
{

/// Return sin and cos of phi as complex.

template <typename T>
inline vsip::complex<T>
sin_cos(double phi)
{
  return vsip::complex<T>(cos(phi), sin(phi));
}



// Reference 1-D DFT algorithm.  Brutes it out, but easy to validate
// and works for any size.

// Requires:
//   IN to be input Vector.
//   OUT to be output Vector, of same size as IN.
//   IDIR to be sign of exponential.
//     -1 => Forward Fft,
//     +1 => Inverse Fft.

template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
void dft(
  vsip::const_Vector<T1, Block1> in,
  vsip::Vector<T2, Block2>       out,
  int                            idir)
{
  using vsip::length_type;
  using vsip::index_type;

  length_type const size = in.size();
  assert((sizeof(T1) <  sizeof(T2) && in.size()/2 + 1 == out.size()) ||
	 (sizeof(T1) == sizeof(T2) && in.size() == out.size()));
  typedef double AccT;

  AccT const phi = idir * 2.0 * VSIP_IMPL_PI/size;

  for (index_type w=0; w<out.size(); ++w)
  {
    vsip::complex<AccT> sum = vsip::complex<AccT>();
    for (index_type k=0; k<in.size(); ++k)
      sum += vsip::complex<AccT>(in(k)) * sin_cos<AccT>(phi*k*w);
    out.put(w, T2(sum));
  }
}



// Reference 1-D multi-DFT algorithm on rows of a matrix.

// Requires:
//   IN to be input Matrix.
//   OUT to be output Matrix, of same size as IN.
//   IDIR to be sign of exponential.
//     -1 => Forward Fft,
//     +1 => Inverse Fft.

template <typename T,
	  typename Block1,
	  typename Block2>
void dft_x(
  vsip::Matrix<vsip::complex<T>, Block1> in,
  vsip::Matrix<vsip::complex<T>, Block2> out,
  int                                    idir)
{
  test_assert(in.size(0) == out.size(0));
  test_assert(in.size(1) == out.size(1));
  test_assert(in.local().size(0) == out.local().size(0));
  test_assert(in.local().size(1) == out.local().size(1));

  for (vsip::index_type r=0; r < in.local().size(0); ++r)
    dft(in.local().row(r), out.local().row(r), idir);
}



// Reference 1-D multi-DFT algorithm on columns of a matrix.

template <typename T,
	  typename Block1,
	  typename Block2>
void dft_y(
  vsip::Matrix<vsip::complex<T>, Block1> in,
  vsip::Matrix<vsip::complex<T>, Block2> out,
  int                                    idir)
{
  test_assert(in.size(0) == out.size(0));
  test_assert(in.size(1) == out.size(1));
  test_assert(in.local().size(0) == out.local().size(0));
  test_assert(in.local().size(1) == out.local().size(1));

  for (vsip::index_type c=0; c < in.local().size(1); ++c)
    dft(in.local().col(c), out.local().col(c), idir);
}


template <typename T,
	  typename Block1,
	  typename Block2>
void dft_y_real(
  vsip::Matrix<T, Block1> in,
  vsip::Matrix<vsip::complex<T>, Block2> out)
{
  test_assert(in.size(0)/2 + 1 == out.size(0));
  test_assert(in.size(1) == out.size(1));
  test_assert(in.local().size(0)/2 + 1 == out.local().size(0));
  test_assert(in.local().size(1) == out.local().size(1));

  for (vsip::index_type c=0; c < in.local().size(1); ++c)
    dft(in.local().col(c), out.local().col(c), -1);
}

} // namespace ref
} // namespace vsip_csl

#endif // VSIP_CSL_REF_DFT_HPP

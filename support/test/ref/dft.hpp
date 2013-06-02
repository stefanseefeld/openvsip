//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef test_ref_dft_hpp_
#define test_ref_dft_hpp_

#include <vsip/complex.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>

namespace test
{
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
void dft(vsip::const_Vector<T1, Block1> in,
	 vsip::Vector<T2, Block2>       out,
	 int                            idir)
{
  using vsip::length_type;
  using vsip::index_type;

  length_type const size = in.size();
  assert((sizeof(T1) <  sizeof(T2) && in.size()/2 + 1 == out.size()) ||
	 (sizeof(T1) == sizeof(T2) && in.size() == out.size()));
  typedef double AccT;

  AccT const phi = idir * 2.0 * OVXX_PI/size;

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
void dft_x(vsip::Matrix<vsip::complex<T>, Block1> in,
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
void dft_y(vsip::Matrix<vsip::complex<T>, Block1> in,
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
void dft_y_real(vsip::Matrix<T, Block1> in,
		vsip::Matrix<vsip::complex<T>, Block2> out)
{
  test_assert(in.size(0)/2 + 1 == out.size(0));
  test_assert(in.size(1) == out.size(1));
  test_assert(in.local().size(0)/2 + 1 == out.local().size(0));
  test_assert(in.local().size(1) == out.local().size(1));

  for (vsip::index_type c=0; c < in.local().size(1); ++c)
    dft(in.local().col(c), out.local().col(c), -1);
}

} // namespace test::ref
} // namespace test

#endif

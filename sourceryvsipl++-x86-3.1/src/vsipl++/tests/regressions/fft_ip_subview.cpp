/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/** @file    tests/regressions/fft_ip_subview.cpp
    @author  Jules Bergmann
    @date    2007-07-29
    @brief   VSIPL++ Library: Regression test for in-place Fft of a subview.

    When planning for an in-place split complex FFT, FFTW 3.1.2
    erroneously encodes the gap between the real and imaginary parts.

    When the plan is later applied to another split-complex vector
    with a different gap between real and complex, FFTW will read/write
    memory outside of the vector.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#endif

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

// 1-dim by-reference in-place Fft, complex vector input from a subview.

template <typename T>
void
test_fft_ip_subview()
{
  typedef Fft<const_Vector, T, T, fft_fwd, by_reference, 1, alg_space>
	fft_type;

  length_type rows = 32;
  length_type size = 64;

  index_type r = rows-1;

  fft_type fft(Domain<1>(size), 1.f);

  Vector<T> vec(size);
  Matrix<T> mat(rows, size);

  vec        = T(1, -2);
  mat.row(r) = vec;

  fft(vec);
  fft(mat.row(r));

  test_assert(equal(vec.get(0), T(size, -2*(int)size)));
  for (index_type i=1; i<size; ++i)
    test_assert(equal(vec.get(i), T(0)));

  test_assert(equal(mat.row(r).get(0), T(size, -2*(int)size)));
  for (index_type i=1; i<size; ++i)
    test_assert(equal(mat.row(r).get(i), T(0)));
}



// By-reference Fftm, in-place

template <typename T>
void
test_fftm_ip_subview(
  bool        scale)
{
  typedef Fftm<T, T, row, fft_fwd, by_reference, 1> fftm_type;

  length_type rows = 16;
  length_type cols = 64;

  Matrix<T> inout(rows, cols,         T(100, -1));
  Matrix<T> big_inout(2*rows, 2*cols, T(-101));

  Domain<2> dom(Domain<1>(rows/2, 1, rows), Domain<1>(cols/2, 1, cols));

  fftm_type fftm(Domain<2>(rows, cols), scale ? 1.f / cols : 1.f);

  for (index_type r=0; r<rows; ++r)
  {
    inout.row(r) = T(r);
    big_inout(dom).row(r) = T(r);
  }

  fftm(inout); 
  fftm(big_inout(dom)); 

  for (index_type r=0; r<rows; ++r)
  {
#if VERBOSE
    if (!(inout.get(r, 0) == T(scale ? r : r*cols)))
    {
      cout << "test_fftm_br_ip: miscompare for row " << r << endl
	   << "  expected: " << T(scale ? r : r*cols) << endl
	   << "  got     : " << inout.get(r, 0) << endl
	   << "  scale   : " << (scale ? "true" : "false") << endl;
    }
#endif
    test_assert(inout.get(r, 0) == T(scale ? r : r*cols));
    test_assert(big_inout(dom).get(r, 0) == T(scale ? r : r*cols));
  }
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_fft_ip_subview<complex<float> >();
  test_fftm_ip_subview<complex<float> >(true);

  return 0;
}

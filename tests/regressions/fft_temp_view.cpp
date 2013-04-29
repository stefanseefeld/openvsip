/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/regr_fft_temp_view.cpp
    @author  Jules Bergmann
    @date    2005-09-17
    @brief   VSIPL++ Library: Regression test for FFTs/FFTMs into
	     temporary view.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <cassert>

#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
void
test_fft(length_type size)
{
  Vector<T>   A(size, T());
  Vector<T>   Z(size);

  int const no_times = 1;

  typedef Fft<const_Vector, T, T, fft_fwd, by_reference, no_times, alg_time>
      fft_type;

  fft_type fft(Domain<1>(size), 1.f);

  A = T(1);

  // Note: Only a basic check for correctness.  Original regression
  // was a compilation error.

    
  // FFT: view -> view
  Z = T();
  fft(A, Z);
  test_assert(equal(Z(0), T(size)));

  // FFT: view -> temporary view
  Z = T();
  fft(A, Z(Domain<1>(size)));
  test_assert(equal(Z(0), T(size)));

  // FFT: temporary view -> view
  Z = T();
  fft(A(Domain<1>(size)), Z);
  test_assert(equal(Z(0), T(size)));

  // FFT: temporary view -> temporary view
  Z = T();
  fft(A(Domain<1>(size)), Z(Domain<1>(size)));
  test_assert(equal(Z(0), T(size)));


  // FFT: in-place into view
  Z = A;
  fft(Z);
  test_assert(equal(Z(0), T(size)));

  // FFT: in-place into temporary view
  Z = A;
  fft(Z(Domain<1>(size)));
  test_assert(equal(Z(0), T(size)));
}



template <int      SD,
	  typename T>
void
test_fftm(length_type rows, length_type cols)
{
  Matrix<T>   A(rows, cols, T());
  Matrix<T>   Z(rows, cols);

  int const no_times = 1;

  typedef Fftm<T, T, SD, fft_fwd, by_reference, no_times, alg_time>
      fftm_type;

  fftm_type fftm(Domain<2>(rows, cols), 1.f);

  A = T(1);

  // Note: Only a basic check for correctness.  Original regression
  // was a compilation error.
    
  // FFTM: view -> view
  Z = T();
  fftm(A, Z);
  test_assert(equal(Z(0, 0), SD == 0 ? T(cols) : T(rows)));

  // FFTM: view -> temp view
  Z = T();
  fftm(A, Z(Domain<2>(rows, cols)));
  test_assert(equal(Z(0, 0), SD == 0 ? T(cols) : T(rows)));

  // FFTM: view -> temp view
  Z = T();
  fftm(A(Domain<2>(rows, cols)), Z);
  test_assert(equal(Z(0, 0), SD == 0 ? T(cols) : T(rows)));

  // FFTM: temp view -> temp view
  Z = T();
  fftm(A(Domain<2>(rows, cols)), Z(Domain<2>(rows, cols)));
  test_assert(equal(Z(0, 0), SD == 0 ? T(cols) : T(rows)));


  // FFTM: in-place view
  Z = A;
  fftm(Z);
  test_assert(equal(Z(0, 0), SD == 0 ? T(cols) : T(rows)));

  // FFTM: in-place temporary view
  Z = A;
  fftm(Z(Domain<2>(rows, cols)));
  test_assert(equal(Z(0, 0), SD == 0 ? T(cols) : T(rows)));
}




int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_fft<complex<float> >(32);
  test_fftm<0, complex<float> >(8, 16);
  test_fftm<1, complex<float> >(8, 16);
}

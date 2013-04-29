//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

// Set to 1 to enable verbose output.
#define VERBOSE     0
// Set to 0 to disble use of random values.
#define FILL_RANDOM 1

#include <cmath>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>

#include <vsip/core/config.hpp>
#include <vsip/core/metaprogramming.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/error_db.hpp>
#include <vsip_csl/ref_dft.hpp>

#if VERBOSE
#  include <iostream>
#  include <vsip_csl/output.hpp>
#  include "extdata-output.hpp"
#endif

#include "fft_common.hpp"


#if defined(VSIP_IMPL_FFTW3) || defined(VSIP_IMPL_SAL_FFT)
#  define TEST_2D_CC 1
#endif

#if defined(VSIP_IMPL_FFTW3) || defined(VSIP_IMPL_SAL_FFT)
#  define TEST_2D_RC 1
#endif

#if defined(VSIP_IMPL_FFTW3)
#  define TEST_3D_CC 1
#endif

#  define TEST_3D_RC 0

#if defined(VSIP_IMPL_FFTW3) || defined(VSIP_IMPL_IPP_FFT)
#  define TEST_NON_POWER_OF_2 1
#endif



/***********************************************************************
  Definitions
***********************************************************************/

using namespace std;
using namespace vsip;




// Setup input data for Fft.

template <typename T,
	  typename Block>
void
setup_ptr(int set, Vector<T, Block> in, float scale = 1)
{
  length_type const N = in.size();

  switch(set)
  {
  default:
  case 0:
    in    = T();
    break;
  case 1:
    in    = T();
    in(0) = T(scale);
    break;
  case 2:
    in    = T();
    in(0) = T(1);
    if (N >  1) in(Domain<1>(0, 1, N))    += T(3);
    if (N >  4) in(Domain<1>(0, 4, N/4))  += T(-2);
    if (N > 13) in(Domain<1>(0, 13, N/13)) += T(7);
    if (N > 27) in(Domain<1>(0, 27, N/27)) += T(-15);
    if (N > 37) in(Domain<1>(0, 37, N/37)) += T(31);
    break;
  case 3:
    in    = T(scale);
    break;
  }
}



// check 2D, 3D

template <typename T>
void
test_nd()
{
#if TEST_2D_CC
  test_fft<0,0,complex<T>, complex<T> ,2,vsip::fft_fwd>();
#endif

#if TEST_2D_RC
  test_fft<0,0,T,complex<T>,2,1>();
  test_fft<0,0,T,complex<T>,2,0>();
#endif

#if TEST_3D_CC
  test_fft<0,0,complex<T>,complex<T>,3,vsip::fft_fwd>();
#endif

#if TEST_3D_RC
  test_fft<0,0,T,complex<T>,3,2>();
  test_fft<0,0,T,complex<T>,3,1>();
  test_fft<0,0,T,complex<T>,3,0>();
#endif
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // show_config();

//
// check 2D, 3D
//

#if VSIP_IMPL_PROVIDE_FFT_FLOAT
  test_nd<float>();
#endif 

#if VSIP_IMPL_PROVIDE_FFT_DOUBLE && VSIP_IMPL_TEST_DOUBLE
  test_nd<double>();
#endif

#if VSIP_IMPL_PROVIDE_FFT_LONG_DOUBLE
  test_nd<long double>();
#endif

  return 0;
}

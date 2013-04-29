/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/fft_block_type.cpp
    @author  Jules Bergmann
    @date    2005-06-17
    @brief   VSIPL++ Library: Testcases for Fft with different block types.
*/

/***********************************************************************
  Included Files
***********************************************************************/

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



/***********************************************************************
  Definitions
***********************************************************************/

using namespace std;
using namespace vsip;



// Check with different block types.

template <typename T>
void
test_block_type()
{
#if TEST_2D_CC
  test_fft<0,1,complex<T>,complex<T>,2,vsip::fft_fwd>();
  test_fft<1,0,complex<T>,complex<T>,2,vsip::fft_fwd>();

#  if VSIP_IMPL_TEST_LEVEL > 0
  test_fft<0,2,complex<T>,complex<T>,2,vsip::fft_fwd>();
  test_fft<1,1,complex<T>,complex<T>,2,vsip::fft_fwd>();
  test_fft<1,2,complex<T>,complex<T>,2,vsip::fft_fwd>();
  test_fft<2,0,complex<T>,complex<T>,2,vsip::fft_fwd>();
  test_fft<2,1,complex<T>,complex<T>,2,vsip::fft_fwd>();
  test_fft<2,2,complex<T>,complex<T>,2,vsip::fft_fwd>();
#  endif
#endif

#if TEST_2D_RC
  test_fft<0,1,T,complex<T>,2,1>();
  test_fft<0,1,T,complex<T>,2,0>();
  test_fft<1,0,T,complex<T>,2,1>();
  test_fft<1,0,T,complex<T>,2,0>();

#  if VSIP_IMPL_TEST_LEVEL > 0
  test_fft<0,2,T,complex<T>,2,1>();
  test_fft<0,2,T,complex<T>,2,0>();

  test_fft<1,1,T,complex<T>,2,1>();
  test_fft<1,1,T,complex<T>,2,0>();
  test_fft<1,2,T,complex<T>,2,1>();
  test_fft<1,2,T,complex<T>,2,0>();

  test_fft<2,0,T,complex<T>,2,1>();
  test_fft<2,0,T,complex<T>,2,0>();
  test_fft<2,1,T,complex<T>,2,1>();
  test_fft<2,1,T,complex<T>,2,0>();
  test_fft<2,2,T,complex<T>,2,1>();
  test_fft<2,2,T,complex<T>,2,0>();
#  endif
#endif

#if TEST_3D_CC
  test_fft<0,1,complex<T>,complex<T>,3,vsip::fft_fwd>();
  test_fft<1,0,complex<T>,complex<T>,3,vsip::fft_fwd>();

#  if VSIP_IMPL_TEST_LEVEL > 0
  test_fft<0,2,complex<T>,complex<T>,3,vsip::fft_fwd>();
  test_fft<1,1,complex<T>,complex<T>,3,vsip::fft_fwd>();
  test_fft<1,2,complex<T>,complex<T>,3,vsip::fft_fwd>();
  test_fft<2,0,complex<T>,complex<T>,3,vsip::fft_fwd>();
  test_fft<2,1,complex<T>,complex<T>,3,vsip::fft_fwd>();
  test_fft<2,2,complex<T>,complex<T>,3,vsip::fft_fwd>();
#  endif
#endif

#if TEST_3D_RC
  test_fft<0,1,T,complex<T>,3,2>();
  test_fft<0,1,T,complex<T>,3,1>();
  test_fft<0,1,T,complex<T>,3,0>();
  test_fft<1,0,T,complex<T>,3,2>();
  test_fft<1,0,T,complex<T>,3,1>();
  test_fft<1,0,T,complex<T>,3,0>();

#  if VSIP_IMPL_TEST_LEVEL > 0
  test_fft<0,2,T,complex<T>,3,2>();
  test_fft<0,2,T,complex<T>,3,1>();
  test_fft<0,2,T,complex<T>,3,0>();

  test_fft<1,1,T,complex<T>,3,2>();
  test_fft<1,1,T,complex<T>,3,1>();
  test_fft<1,1,T,complex<T>,3,0>();
  test_fft<1,2,T,complex<T>,3,2>();
  test_fft<1,2,T,complex<T>,3,1>();
  test_fft<1,2,T,complex<T>,3,0>();

  test_fft<2,0,T,complex<T>,3,2>();
  test_fft<2,0,T,complex<T>,3,1>();
  test_fft<2,0,T,complex<T>,3,0>();
  test_fft<2,1,T,complex<T>,3,2>();
  test_fft<2,1,T,complex<T>,3,1>();
  test_fft<2,1,T,complex<T>,3,0>();
  test_fft<2,2,T,complex<T>,3,2>();
  test_fft<2,2,T,complex<T>,3,1>();
  test_fft<2,2,T,complex<T>,3,0>();
#  endif
#endif
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // show_config();

//
// check with different block types
//
  test_block_type<float>();

  return 0;
}

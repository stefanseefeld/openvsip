/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.
*/

/** @file    tests/plugin/fft.cpp
    @author  Jules Bergmann
    @date    2009-06-02
    @brief   VSIPL++ Library: Test plugin Fft sizes.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <algorithm>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

// 1-dim by-value Fft, out-of-place

template <typename T>
void
test_fft_bv_op(length_type size, bool scale)
{
  typedef Fft<const_Vector, T, T, fft_fwd, by_value, 1, alg_space>
	fft_type;

  fft_type fft(Domain<1>(size), scale ? 1.f / size : 1.f);

  Vector<T> in(size);
  Vector<T> out(size, T(-100));

  in = T(1);

  out = fft(in); 
  test_assert(out.get(0) == T(scale ? 1 : size));

  const_Vector<T> c_in(in);
  out = T(-101);
  out = fft(c_in); 

  test_assert(out.get(0) == T(scale ? 1 : size));
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  for (int i=0; i<4; ++i)
  {
    // Plugins support sizes 4 through 8192, test across that range
    for (length_type size=4; size <= 16384; size *= 2)
      test_fft_bv_op<complex<float> >(size, true);
  }

  return 0;
}

/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/** @file    tests/fft_unaligned.cpp
    @author  Jules Bergmann
    @date    2007-03-16
    @brief   VSIPL++ Library: Test Fft on unaligned views.
*/

/***********************************************************************
  Included Files
***********************************************************************/

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

// Test FFT by-reference out-of-place with given alignment.

template <typename T,
	  storage_format_type C>
void
test_fft_op_align(length_type size, length_type align)
{
  static pack_type const sud_type = dense;
  typedef Layout<1, row1_type, sud_type, C> lp_type;

  typedef impl::Strided<1, T, lp_type> block_type;

  typedef Fft<const_Vector, T, T, fft_fwd, by_reference, 1, alg_space>
	fft_type;

  fft_type fft(Domain<1>(size), 1.f);

  Vector<T, block_type> in (size + align);
  Vector<T, block_type> out(size + align);

  in = T(1);

  fft(in(Domain<1>(align, 1, size)), out(Domain<1>(align, 1, size)));

  test_assert(out.get(align) == T(size));
}



// Test FFT by-reference in-place with given alignment.

template <typename T,
	  storage_format_type C>
void
test_fft_ip_align(length_type size, length_type align)
{
  static pack_type const sud_type = dense;
  typedef Layout<1, row1_type, sud_type, C> lp_type;

  typedef impl::Strided<1, T, lp_type> block_type;

  typedef Fft<const_Vector, T, T, fft_fwd, by_reference, 1, alg_space>
	fft_type;

  fft_type fft(Domain<1>(size), 1.f);

  Vector<T, block_type> inout(size + align);

  inout = T(1);

  fft(inout(Domain<1>(align, 1, size)));

  test_assert(inout.get(align) == T(size));
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_fft_op_align<complex<float>, interleaved_complex>(256, 0);
  test_fft_op_align<complex<float>, interleaved_complex>(256, 1);

  test_fft_ip_align<complex<float>, interleaved_complex>(256, 0);
  test_fft_ip_align<complex<float>, interleaved_complex>(256, 1);

  return 0;
}

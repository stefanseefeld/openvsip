//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <algorithm>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <test.hpp>

using namespace ovxx;

// Test FFT by-reference

template <typename T, storage_format_type SC, storage_format_type DC>
void
test_fft_br(length_type size)
{
  typedef Layout<1, row1_type, dense, SC> src_lp_type;
  typedef Layout<1, row1_type, dense, DC> dst_lp_type;

  typedef Strided<1, T, src_lp_type> src_block_type;
  typedef Strided<1, T, dst_lp_type> dst_block_type;

  typedef Fft<const_Vector, T, T, fft_fwd, by_reference, 1, alg_space>
	fft_type;

  fft_type fft(Domain<1>(size), 1.f);

  Vector<T, src_block_type> in(size);
  Vector<T, dst_block_type> out(size);

  in = T(1);

  fft(in, out);

  test_assert(out.get(0) == T(size));
}



// Test FFT by-value

template <typename T, storage_format_type SC, storage_format_type DC>
void
test_fft_bv(length_type size)
{
  typedef Layout<1, row1_type, dense, SC> src_lp_type;
  typedef Layout<1, row1_type, dense, DC> dst_lp_type;

  typedef Strided<1, T, src_lp_type> src_block_type;
  typedef Strided<1, T, dst_lp_type> dst_block_type;

  typedef Fft<const_Vector, T, T, fft_fwd, by_value, 1, alg_space>
	fft_type;

  fft_type fft(Domain<1>(size), 1.f);

  Vector<T, src_block_type> in(size);
  Vector<T, dst_block_type> out(size);

  in = T(1);

  out = fft(in);

  test_assert(out.get(0) == T(size));
}



// Test FFT by-value in an expression

template <typename T, storage_format_type SC, storage_format_type DC>
void
test_fft_bv_expr(length_type size)
{
  typedef Layout<1, row1_type, dense, SC> src_lp_type;
  typedef Layout<1, row1_type, dense, DC> dst_lp_type;

  typedef Strided<1, T, src_lp_type> src_block_type;
  typedef Strided<1, T, dst_lp_type> dst_block_type;

  typedef Fft<const_Vector, T, T, fft_fwd, by_value, 1, alg_space>
	fft_type;

  fft_type fft(Domain<1>(size), 1.f);

  Vector<T, src_block_type> in(size);
  Vector<T, dst_block_type> out(size);

  in = T(1);
  out = T(0);

  out = out + fft(in);

  test_assert(out.get(0) == T(size));
}


template <typename T>
void
test_set(length_type size)
{
  test_fft_br<complex<float>, interleaved_complex, interleaved_complex>(size);
  test_fft_br<complex<float>, split_complex, split_complex>(size);
  test_fft_br<complex<float>, interleaved_complex, split_complex>(size);
  test_fft_br<complex<float>, split_complex, interleaved_complex>(size);

  test_fft_bv<complex<float>, interleaved_complex, interleaved_complex>(size);
  test_fft_bv<complex<float>, split_complex, split_complex>(size);
  test_fft_bv<complex<float>, interleaved_complex, split_complex>(size);
  test_fft_bv<complex<float>, split_complex, interleaved_complex>(size);

  test_fft_bv_expr<complex<float>, interleaved_complex, interleaved_complex>(size);
  test_fft_bv_expr<complex<float>, split_complex, split_complex>(size);
  test_fft_bv_expr<complex<float>, interleaved_complex, split_complex>(size);
  test_fft_bv_expr<complex<float>, split_complex, interleaved_complex>(size);
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // test_set<complex<float> >(16);
  test_set<complex<float> >(256);

  return 0;
}

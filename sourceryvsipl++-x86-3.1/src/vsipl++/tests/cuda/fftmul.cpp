/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    examples/fconv_cuda.cpp
    @author  Don McCoy
    @date    2009-08-15
    @brief   VSIPL++ Library: CUDA-based test for combined Fft + Multiply
               evaluator.
*/

#include <vsip/initfin.hpp>
#include <vsip/random.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/error_db.hpp>
#include <vsip_csl/output.hpp>

using namespace std;
using namespace vsip;
using namespace vsip::impl;
using namespace vsip::impl::profile;
using namespace vsip_csl;


#ifndef DEBUG
#define DEBUG        0
#endif

template <typename T,
          typename Block>
void
fftmul_test(length_type rows, length_type cols)
{
  Matrix<T, Block> in(rows, cols, T());
  Matrix<T, Block> k(rows, cols, T(2));
  Matrix<T, Block> out(rows, cols, T());
  Matrix<T, Block> ref(rows, cols, T());

#ifndef DEBUG
  Rand<T> gen(0, 0);
  in = gen.randu(rows, cols);
  k = gen.randu(rows, cols);
#else
  in.col(0) = T(cols);
#endif

  typedef Fftm<T, T, row, fft_fwd, by_value> for_fftm_type;
  for_fftm_type for_fftm(Domain<2>(rows, cols), 1.0f/cols);

  // Compute the reference in separate steps
  ref = for_fftm(in);
  ref *= k;

  // Compute output as a combined expression
  out = k * for_fftm(in);


#if DEBUG
  cout << "in:  "  << endl << in.row(0) << endl;
  cout << "out: "  << endl << out.row(0) << endl;
  cout << "ref: "  << endl << out.row(0) << endl;
#endif

  // Verify they match
  test_assert(view_equal(out, ref));
}


int
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  typedef std::complex<float> T;
  typedef vsip::Dense<2, T>  dense_block_type;

#if DEBUG
  fftmul_test<T, dense_block_type>(16, 32);
#else
  fftmul_test<T, dense_block_type>(128, 256);
  fftmul_test<T, dense_block_type>(357, 1253);
  fftmul_test<T, dense_block_type>(1024, 2048);
#endif
  return 0;
}

/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    convolution.cpp
    @author  Jules Bergmann
    @date    2005-06-09
    @brief   VSIPL++ Library: Regression tests for [signal.convolution].

    Regression test: dda::Data based convolution was not writing data
    correctly to subviews with non-unit-stride.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/error_db.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

double const ERROR_THRESH = -100;

length_type expected_output_size(
  support_region_type supp,
  length_type         M,    // kernel length
  length_type         N,    // input  length
  length_type         D)    // decimation factor
{
  if      (supp == support_full)
    return ((N + M - 2)/D) + 1;
  else if (supp == support_same)
    return ((N - 1)/D) + 1;
  else //(supp == support_min)
    return ((N - 1)/D) - ((M-1)/D) + 1;
}



length_type expected_shift(
  support_region_type supp,
  length_type         M,     // kernel length
  length_type         /*D*/) // decimation factor
{
  if      (supp == support_full)
    return 0;
  else if (supp == support_same)
    return (M/2);
  else //(supp == support_min)
    return (M-1);
}



/// Test convolution with nonsym symmetry.

template <typename            T,
	  support_region_type support>
void
test_conv_nonsym_split(
  length_type M,
  index_type  c1,
  index_type  c2,
  int         k1,
  int         k2)
{
  symmetry_type const        symmetry = nonsym;

  typedef Convolution<const_Vector, symmetry, support, T> conv_type;

  // length_type const M = 5;				// filter size
  length_type const N = 100;				// input size
  length_type const D = 1;				// decimation
  length_type const P = expected_output_size(support, M, N, D);
  // ((N-1)/D) - ((M-1)/D) + 1;	// output size

  int shift = expected_shift(support, M, D);

  Vector<T> coeff1(M, T());
  Vector<T> coeff2(M, T());

  coeff1(c1) = T(k1);
  coeff2(c2) = T(k2);

  conv_type conv1(coeff1, Domain<1>(N), D);
  conv_type conv2(coeff2, Domain<1>(N), D);

  test_assert(conv1.symmetry() == symmetry);
  test_assert(conv1.support()  == support);
  test_assert(conv1.kernel_size().size()  == M);
  test_assert(conv1.filter_order().size() == M);
  test_assert(conv1.input_size().size()   == N);
  test_assert(conv1.output_size().size()  == P);

  test_assert(conv2.symmetry() == symmetry);
  test_assert(conv2.support()  == support);
  test_assert(conv2.kernel_size().size()  == M);
  test_assert(conv2.filter_order().size() == M);
  test_assert(conv2.input_size().size()   == N);
  test_assert(conv2.output_size().size()  == P);


  Vector<T> in(N);
  Vector<complex<T> > out(P);
  Vector<complex<T> > exp(P, complex<T>(201, -301));

  for (index_type i=0; i<N; ++i)
    in(i) = T(i);

  conv1(in, out.real());
  conv2(in, out.imag());

  for (index_type i=0; i<P; ++i)
  {
    T val1, val2;

    if ((int)i + shift - (int)c1 < 0 || i + shift - c1 >= in.size())
      val1 = T();
    else
      val1 = in(i + shift - c1);

    if ((int)i + shift - (int)c2 < 0 || i + shift - c2 >= in.size())
      val2 = T();
    else
      val2 = in(i + shift - c2);

    exp.put(i, complex<T>(T(k1) * val1, T(k2) * val2));
  }

  double error = error_db(out, exp);
  test_assert(error < ERROR_THRESH);
}



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_conv_nonsym_split<float, support_min>(4, 0, 1, +1, +1);
  test_conv_nonsym_split<float, support_min>(5, 0, 1, +1, -1);

  test_conv_nonsym_split<float, support_same>(4, 0, 1, +1, +1);
  test_conv_nonsym_split<float, support_same>(5, 0, 1, +1, -1);

  test_conv_nonsym_split<float, support_full>(4, 0, 1, +1, +1);
  test_conv_nonsym_split<float, support_full>(5, 0, 1, +1, -1);
}

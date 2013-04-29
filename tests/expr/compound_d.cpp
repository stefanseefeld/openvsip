//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Tests for compound expressions, double-precision floating point

#include <iostream>
#include <vsip/initfin.hpp>
#include "compound.hpp"

using namespace vsip;


int
main(int argc, char **argv)
{
  vsipl init(argc, argv);
 
  // 8 * 64 * 128 = 64 K
  length_type const M = 8;
  length_type const N = 64;
  length_type const P = 128;

  // These run through cases of 1-, 2- and 3-dimensions, keeping
  // the view size a constant (a product of M, N and P).
  test_elementwise_cases<std::complex<double> >(M, N, P);

  // This test only handles vectors, again we keep the view size
  // a constant (and large enough to be greater than any thresholding
  // value in place for a given backend).  Both real and complex
  // values, including expressions with mixed types are checked.
  test_nonelementwise_vectors<double>(M * N * P);

  return 0;
}

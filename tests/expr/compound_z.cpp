/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Tests for compound expressions, double-precision complex floating point

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

  return 0;
}

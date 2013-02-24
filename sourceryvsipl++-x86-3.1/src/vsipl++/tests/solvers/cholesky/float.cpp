/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/solvers/cholesky/float.cpp
    @author  Jules Bergmann
    @date    2008-10-07
    @brief   VSIPL++ Library: Unit tests for Cholesky solver.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/solvers.hpp>

#include "cholesky.hpp"



/***********************************************************************
  Main
***********************************************************************/

template <> float  Precision_traits<float>::eps = 0.0;



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  Precision_traits<float>::compute_eps();

#if BAD_MATRIX_A
  test_chold_file<float>(upper, "cholesky-bad-float-42.dat", 42, 1);
#endif

  chold_cases<float>           (upper);
  chold_cases<complex<float> > (upper);

  chold_cases<float>           (lower);
  chold_cases<complex<float> > (lower);
}

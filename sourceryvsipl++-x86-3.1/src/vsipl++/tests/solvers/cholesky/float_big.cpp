/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/solvers/cholesky/float_big.cpp
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

  chold_big_cases<float>           (upper);
  chold_big_cases<complex<float> > (upper);

  chold_big_cases<float>           (lower);
  chold_big_cases<complex<float> > (lower);
}

/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/view_functions_binary.hpp
    @author  Stefan Seefeld
    @date    2005-03-16
    @brief   VSIPL++ Library: Unit tests for binary View expressions.

    This file contains unit tests for View expressions.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cassert>
#include <complex>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/dense.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/test.hpp>

#include "view_functions.hpp"

using namespace std;
using namespace vsip;



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  TEST_TERNARY(am, scalar_f, 1.f, 2.f, 3.f)
  TEST_TERNARY(ma, scalar_f, 1.f, 2.f, 3.f)
  TEST_TERNARY(msb, scalar_f, 1.f, 2.f, 3.f)
  TEST_TERNARY(sbm, scalar_f, 1.f, 2.f, 3.f)
}

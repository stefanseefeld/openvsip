/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/scalar_view_add.cpp
    @author  Jules Bergmann
    @date    2007-02-08
    @brief   VSIPL++ Library: Coverage tests for scalar-view add expressions.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>

#include "scalar_view.hpp"



/***********************************************************************
  Definitions
***********************************************************************/

int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  test_lite<op_add>();
#else
  test<op_add>();
#endif
}

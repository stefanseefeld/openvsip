/* Copyright (c) 2006, 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/check_config.cpp
    @author  Jules Bergmann
    @date    2006-10-04
    @brief   VSIPL++ Library: Check library configuration
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/core/check_config.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  cout << vsip::impl::library_config();

  test_assert(vsip::impl::app_config() == vsip::impl::library_config());

  return 0;
}

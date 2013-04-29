/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    support.cpp
    @author  Jules Bergmann
    @date    2005-01-19
    @brief   VSIPL++ Library: Unit tests for [support] items.

    This file has unit tests for functionality defined in the [support]
    section of the VSIPL++ specification.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <vsip/support.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

/// Test throw and catch of computation_error exception.
void
test_computation_error()
{
#if VSIP_HAS_EXCEPTIONS
  int pass = 0;

  try
  {
    VSIP_THROW(computation_error("TEST: throw exception"));
    test_assert(0);
  }
  catch (const std::exception& error)
  {
    // cout << "Caught: " << error.what() << endl;
    if (error.what() == std::string("TEST: throw exception"))
      pass = 1;
  }

  test_assert(pass);
#else
  // Could report untested or not-applicable.
#endif
}



int
main()
{
  test_computation_error();
}

//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <vsip/support.hpp>
#include <test.hpp>

using namespace ovxx;

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

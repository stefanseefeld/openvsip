//
// Copyright (c) 2006, 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

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

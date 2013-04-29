//
// Copyright (c) 2006, 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>

#include <vsip/initfin.hpp>
#include <ovxx/support.hpp>
#include <ovxx/check_config.hpp>
#include <test.hpp>

using namespace ovxx;

int main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);
  std::cout << library_config();
  test_assert(app_config() == library_config());
}

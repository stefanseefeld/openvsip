//
// Copyright (c) 2015 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "block_api.hpp"

BOOST_PYTHON_MODULE(_block_d)
{
  using namespace pyvsip;
  initialize();
  define_distributed_block<1, double, Map<> >("block1");
  define_distributed_block<2, double, Map<> >("block2");
}

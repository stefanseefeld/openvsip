//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "block_api.hpp"

BOOST_PYTHON_MODULE(_block_f)
{
  using namespace pyvsip;
  initialize();
  define_distributed_block<1, float, Map<> >("block1");
  define_distributed_block<2, float, Map<> >("block2");
}

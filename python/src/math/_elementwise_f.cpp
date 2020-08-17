//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "elementwise.hpp"

BOOST_PYTHON_MODULE(_elementwise_f)
{
  using namespace pyvsip;
  initialize();
  define_elementwise<1, float>();
  define_elementwise<2, float>();
}

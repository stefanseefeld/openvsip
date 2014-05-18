//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "dda_api.hpp"

BOOST_PYTHON_MODULE(fdda)
{
  using namespace pyvsip;
  define_dda<1, float>("dda1");
  define_dda<2, float>("dda2");
}

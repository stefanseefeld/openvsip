//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "matvec_api.hpp"

BOOST_PYTHON_MODULE(_matvec_cf)
{
  using namespace pyvsip;
  initialize();
  define_api<complex<float> >();
}

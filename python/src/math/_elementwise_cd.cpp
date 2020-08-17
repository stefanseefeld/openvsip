//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "elementwise.hpp"

BOOST_PYTHON_MODULE(_elementwise_cd)
{
  using namespace pyvsip;
  initialize();
  define_complex_elementwise<1, complex<double> >();
  define_complex_elementwise<2, complex<double> >();
}

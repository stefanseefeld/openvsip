//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "elementwise.hpp"

BOOST_PYTHON_MODULE(_elementwise_cf)
{
  using namespace pyvsip;
  import_array();
  // initialize();
  define_complex_elementwise<1, complex<float> >();
  define_complex_elementwise<2, complex<float> >();
}

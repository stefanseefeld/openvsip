//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "fft.hpp"

BOOST_PYTHON_MODULE(_fft_cf)
{
  using namespace pyvsip;
  define_complex_fft<float>();
}

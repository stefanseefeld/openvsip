//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "corr.hpp"

BOOST_PYTHON_MODULE(_corr_cd)
{
  using namespace pyvsip;
  define_corr<vsip::complex<double> >();
}

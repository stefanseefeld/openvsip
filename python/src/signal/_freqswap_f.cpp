//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "freqswap.hpp"

BOOST_PYTHON_MODULE(_freqswap_f)
{
  using namespace pyvsip;
  bpl::def("freqswap", freqswap<1, float>);
  bpl::def("freqswap", freqswap<2, float>);
}

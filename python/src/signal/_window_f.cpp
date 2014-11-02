//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "window.hpp"

BOOST_PYTHON_MODULE(_window_f)
{
  using namespace pyvsip;

  bpl::def("blackman", blackman<float>);
  bpl::def("cheby", cheby<float>);
  bpl::def("hanning", hanning<float>);
  bpl::def("kaiser", kaiser<float>);

}

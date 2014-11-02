//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "solvers.hpp"

BOOST_PYTHON_MODULE(_llsqsol_d)
{
  using namespace pyvsip;
  bpl::def("llsqsol", llsqsol<double>);
}

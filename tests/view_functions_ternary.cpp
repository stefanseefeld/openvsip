//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <cassert>
#include <complex>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/dense.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/test.hpp>

#include "view_functions.hpp"

using namespace std;
using namespace vsip;



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  TEST_TERNARY(am, scalar_f, 1.f, 2.f, 3.f)
  TEST_TERNARY(ma, scalar_f, 1.f, 2.f, 3.f)
  TEST_TERNARY(msb, scalar_f, 1.f, 2.f, 3.f)
  TEST_TERNARY(sbm, scalar_f, 1.f, 2.f, 3.f)
}

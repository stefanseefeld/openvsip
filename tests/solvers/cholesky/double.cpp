//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/solvers.hpp>
#include "cholesky.hpp"

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test::precision<double>::init();

  chold_cases<double>          (upper);
  chold_cases<complex<double> >(upper);
  chold_cases<double>          (lower);
  chold_cases<complex<double> >(lower);
}

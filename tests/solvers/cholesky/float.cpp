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



/***********************************************************************
  Main
***********************************************************************/

template <> float  Precision_traits<float>::eps = 0.0;



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  Precision_traits<float>::compute_eps();

#if BAD_MATRIX_A
  test_chold_file<float>(upper, "cholesky-bad-float-42.dat", 42, 1);
#endif

  chold_cases<float>           (upper);
  chold_cases<complex<float> > (upper);

  chold_cases<float>           (lower);
  chold_cases<complex<float> > (lower);
}

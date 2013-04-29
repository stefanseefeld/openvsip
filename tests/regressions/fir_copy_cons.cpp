//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/test.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
void
test_fir_cc()
{
  using vsip::length_type;
  using vsip::nonsym;
  using vsip::state_save;
  using vsip::Fir;
  using vsip::Vector;

  length_type N = 4;
  length_type M = 2;
  length_type D = 1;

  Vector<T> kernel(M, T(1));
  Vector<T> input (N, T(1));
  Vector<T> output(N, T(-101));

  Fir<T,nonsym,state_save,1>  fir1(kernel, N, D);

  // Frame 1 -------------------------------------------------

  fir1(input, output);

  test_assert(output(0) == T(1));
  test_assert(output(1) == T(2));
  test_assert(output(2) == T(2));
  test_assert(output(3) == T(2));

  // Frame 2 -------------------------------------------------

  fir1(input, output);

  test_assert(output(0) == T(2));
  test_assert(output(1) == T(2));
  test_assert(output(2) == T(2));
  test_assert(output(3) == T(2));

  // Frame 3 -------------------------------------------------

  Fir<T,nonsym,state_save,1>  fir2(fir1);

  // fir2 should have the same saved state as fir1.

  fir2(input, output);

  test_assert(output(0) == T(2));
  test_assert(output(1) == T(2));
  test_assert(output(2) == T(2));
  test_assert(output(3) == T(2));
}



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);
  test_fir_cc<float>();
}

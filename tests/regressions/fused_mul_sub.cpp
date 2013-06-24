//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <test.hpp>

using namespace ovxx;

template <typename T>
void
run_test()
{
  length_type size = 4;

  Vector<T> A(size, T(3));
  T         b = T(4);
  Vector<T> C(size, T(3));
  Vector<T> Z(size, T());

  // --------------------------------------------------------------------
  // SAL dispatch had this going to b*C - A instead of A - b*C.
  Z = A - b*C;

  for (index_type i=0; i<size; ++i)
    test_assert(Z.get(i) == A.get(i) - b*C.get(i));

  // --------------------------------------------------------------------
  Z = A + b*C;

  for (index_type i=0; i<size; ++i)
    test_assert(Z.get(i) == A.get(i) + b*C.get(i));

  // --------------------------------------------------------------------
  Z = b * (A+C);

  for (index_type i=0; i<size; ++i)
    test_assert(Z.get(i) == b*(A.get(i) + C.get(i)));

  // --------------------------------------------------------------------
  // SAL dispatch wasn't catching this
  // (wrong order of args to is_op_supported)
  Z = b * (A-C);

  for (index_type i=0; i<size; ++i)
    test_assert(Z.get(i) == b*(A.get(i) - C.get(i)));
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  run_test<float>();
  run_test<double>();
  run_test<complex<float> >();
  run_test<complex<double> >();

  return 0;
}

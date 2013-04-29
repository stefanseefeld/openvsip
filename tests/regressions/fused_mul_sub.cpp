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

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
void
test()
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



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test<float>();
  test<double>();
  test<complex<float> >();
  test<complex<double> >();

  return 0;
}

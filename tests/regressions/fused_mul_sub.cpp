/* Copyright (c) 2007 by CodeSourcery, LLC.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/regressions/fused_mul_sub.cpp
    @author  Jules Bergmann
    @date    2007-08-23
    @brief   VSIPL++ Library: Regresions for cases mishandled by SAL dispatch.
*/

/***********************************************************************
  Included Files
***********************************************************************/

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

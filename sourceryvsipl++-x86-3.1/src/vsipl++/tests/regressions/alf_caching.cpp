/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/regressions/alf_caching.cpp
    @author  Jules Bergmann
    @date    2008-07-01
    @brief   VSIPL++ Library: Test switching between CML and VSIPL++
             ALF tasks.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/opt/assign_diagnostics.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

template <typename T>
void
test(length_type size)
{
  length_type loop = 10;

  Matrix<T> M(size, size, T(0)); M.diag() = T(2);
  Vector<T> a(size,    T(3));
  Vector<T> b(size,    T(4));
  Vector<T> c(size,    T(5));
  Vector<T> d(size,    T(5));

#if DEBUG
  vsip::impl::assign_diagnostics(c, a * b);
#endif

  for (index_type l=0; l<loop; ++l)
  {
    c = T(0);
    d = T(0);

    d = prod(M, a);
    test_assert(d.get(0) == T(2*3));

    c = a * b;
    test_assert(c.get(0) == T(3*4));
  }
}



int
main(int argc, char** argv)
{
  typedef complex<float> Cf;

  vsipl init(argc, argv);

  test<float>(128*16);
}

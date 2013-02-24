/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    tests/ip_transpose.cpp
    @author  Jules Bergmann
    @date    2008-06-16
    @brief   VSIPL++ Library: Unit tests for in-place transpose.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/diagnostics.hpp>
#include <iostream>

using namespace vsip;


/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

template <typename T>
void
ip_transpose(length_type size)
{
  Matrix<T> A(size, size);

  for (index_type r=0; r<size; ++r)
    for (index_type c=0; c<size; ++c)
      A.put(r, c, T(r*size+c));

  vsip_csl::assign_diagnostics(A, A.transpose());
  A = A.transpose();

  for (index_type r=0; r<size; ++r)
    for (index_type c=0; c<size; ++c)
      test_assert(A.get(r, c) == T(c*size+r));
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  ip_transpose<float>(32);
  ip_transpose<complex<float> >(32);
}

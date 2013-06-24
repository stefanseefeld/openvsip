//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <ovxx/assign/diagnostics.hpp>
#include <test.hpp>

using namespace ovxx;

template <typename T>
void
ip_transpose(length_type size)
{
  Matrix<T> A(size, size);

  for (index_type r=0; r<size; ++r)
    for (index_type c=0; c<size; ++c)
      A.put(r, c, T(r*size+c));

  assignment::diagnostics(A, A.transpose());
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

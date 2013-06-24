//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>
#include <test.hpp>

using namespace ovxx;

template <typename T>
void
test_assign(Domain<2> const& dom)
{
  length_type const rows = dom[0].size();
  length_type const cols = dom[1].size();

  Matrix<T, Dense<2, T, row2_type> > src(rows, cols);
  Matrix<T, Dense<2, T, col2_type> > dst1(rows, cols, T());
  Matrix<T, Dense<2, T, col2_type> > dst2(rows, cols, T());

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
      src(r, c) = T(r*cols+c);

  // Setup transpose so that RHS has negative strides for both
  // rows and columns.
  dst1 = src(Domain<2>(Domain<1>(rows-1, -1, rows),
		      Domain<1>(cols-1, -1, cols)));
  dst2(Domain<2>(Domain<1>(rows-1, -1, rows), Domain<1>(cols-1, -1, cols)))
    = src;

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
    {
      index_type rr = rows-1-r;
      index_type cc = cols-1-c;
      test_assert(equal(dst1(r, c), T(rr*cols+cc)));
      test_assert(equal(dst1(r, c), src(rr, cc)));

      test_assert(equal(dst2(r, c), T(rr*cols+cc)));
      test_assert(equal(dst2(r, c), src(rr, cc)));
    }
}


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // These tests caused seg-faults in fast-transpose on
  // systems where sizeof(unsigned) != sizeof(stride_type),
  // such as x86-64.


  test_assign<float>(Domain<2>(3, 3));
  test_assign<float>(Domain<2>(3, 4));
  test_assign<float>(Domain<2>(16, 17));

  test_assign<complex<float> >(Domain<2>(3, 3));
  test_assign<complex<float> >(Domain<2>(4, 8));
  test_assign<complex<float> >(Domain<2>(16, 64));
  test_assign<complex<float> >(Domain<2>(64, 32));
  test_assign<complex<float> >(Domain<2>(256, 256));
}

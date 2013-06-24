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

// Test transpose between arrays with non-unit-stride

template <typename T>
void
test_assign(Domain<2> const& dom)
{
  length_type const rows = dom[0].size();
  length_type const cols = dom[1].size();

  typedef typename Matrix<T, Dense<2, T, row2_type> >::subview_type src_view;
  typedef typename Matrix<T, Dense<2, T, col2_type> >::subview_type dst_view;

  Matrix<T, Dense<2, T, row2_type> > big_src(2*rows, 2*cols, T(-5));
  Matrix<T, Dense<2, T, col2_type> > big_dst(2*rows, 2*cols, T(-10));

  src_view src = big_src(Domain<2>(Domain<1>(0, 2, rows),
				   Domain<1>(0, 2, cols)));
  dst_view dst = big_dst(Domain<2>(Domain<1>(0, 2, rows),
				   Domain<1>(0, 2, cols)));


  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
      src(r, c) = T(r*cols+c);

  dst = src;

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
    {
      test_assert(equal(dst(r, c), T(r*cols+c)));
      test_assert(equal(dst(r, c), src(r, c)));
    }
}


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_assign<float>(Domain<2>(16, 17));
  test_assign<float>(Domain<2>(32, 16));

  test_assign<complex<float> >(Domain<2>(16, 64));
  test_assign<complex<float> >(Domain<2>(64, 32));
  test_assign<complex<float> >(Domain<2>(256, 256));
}

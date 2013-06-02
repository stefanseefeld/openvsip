//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <cassert>
#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include "subviews.hpp"

using namespace vsip;

template <typename T,
	  typename OrderT>
void
matrix_test(Domain<2> const& dom)
{
  typedef Dense<2, T, OrderT>           block_type;
  typedef Matrix<T, block_type>         view_type;
  view_type mat(dom[0].size(), dom[1].size());

  test_matrix(mat);
}


template <typename T>
void
test_for_type()
{
  matrix_test<T, row2_type>(Domain<2>(5, 7));
  matrix_test<T, col2_type>(Domain<2>(5, 7));
}




int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  test_for_type<float>();
#else
  test_for_type<float>();
  test_for_type<complex<float> >();
#endif

  return 0;
}

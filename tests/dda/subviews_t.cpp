/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/extdata_subviews_t.cpp
    @author  Jules Bergmann
    @date    2005-07-22
    @brief   VSIPL++ Library: Unit tests for DDI to tensor subviews.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>

#include "subviews.hpp"

using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

template <typename T,
	  typename OrderT>
void
tensor_test(Domain<3> const& dom)
{
  typedef Dense<3, T, OrderT>           block_type;
  typedef Tensor<T, block_type>         view_type;
  view_type view(dom[0].size(), dom[1].size(), dom[2].size());

  test_tensor(view);
}


template <typename T>
void
test_for_type()
{
  tensor_test<T, tuple<0, 1, 2> >(Domain<3>(5, 7, 9));
#if 0
  tensor_test<T, tuple<0, 2, 1> >(Domain<3>(5, 7, 9));
  tensor_test<T, tuple<1, 0, 2> >(Domain<3>(5, 7, 9));
  tensor_test<T, tuple<1, 2, 0> >(Domain<3>(5, 7, 9));
  tensor_test<T, tuple<2, 0, 1> >(Domain<3>(5, 7, 9));
#endif
  tensor_test<T, tuple<2, 1, 0> >(Domain<3>(5, 7, 9));
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

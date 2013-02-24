/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/extdata_subviews_v.cpp
    @author  Jules Bergmann
    @date    2005-07-22
    @brief   VSIPL++ Library: Unit tests for DDI to vector subviews.
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

template <typename T>
void
vector_test(Domain<1> const& dom)
{
  // Regular Dense
  typedef Dense<1, T>           block_type;
  typedef Vector<T, block_type> view_type;
  view_type view(dom[0].size());

  test_vector(view);
}



template <typename T, storage_format_type C>
void
vector_strided_test(Domain<1> const& dom)
{
  typedef vsip::Layout<1, row1_type, vsip::dense, C> layout_type;
  typedef vsip::impl::Strided<1, T, layout_type> block_type;
  typedef Vector<T, block_type> view_type;

  view_type view(dom[0].size());
  
  test_vector(view);
}



template <typename T>
void
test_for_type()
{
  vector_test<T>(Domain<1>(7));

  vector_strided_test<T, vsip::interleaved_complex>(Domain<1>(7));
  vector_strided_test<T, vsip::split_complex>(Domain<1>(7));
}




int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_TEST_LEVEL == 0
  vector_test<float>(Domain<1>(7));
#else
  test_for_type<float>();
  test_for_type<complex<float> >();
#endif

  return 0;
}

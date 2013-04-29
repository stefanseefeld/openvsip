//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

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

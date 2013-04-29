//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/map.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;



/***********************************************************************
  Definitions
***********************************************************************/

// Test segment_size with PAS algorithm

void
test_pas()
{
  using vsip::impl::segment_size;

  test_assert(segment_size(1, 4, 0) == 1);
  test_assert(segment_size(1, 4, 1) == 0);
  test_assert(segment_size(1, 4, 2) == 0);
  test_assert(segment_size(1, 4, 3) == 0);

  test_assert(segment_size(2, 4, 0) == 1);
  test_assert(segment_size(2, 4, 1) == 1);
  test_assert(segment_size(2, 4, 2) == 0);
  test_assert(segment_size(2, 4, 3) == 0);

  test_assert(segment_size(3, 4, 0) == 1);
  test_assert(segment_size(3, 4, 1) == 1);
  test_assert(segment_size(3, 4, 2) == 1);
  test_assert(segment_size(3, 4, 3) == 0);

  test_assert(segment_size(4, 3, 0) == 2);
  test_assert(segment_size(4, 3, 1) == 2);
  test_assert(segment_size(4, 3, 2) == 0);

  test_assert(segment_size(4, 4, 0) == 1);
  test_assert(segment_size(4, 4, 1) == 1);
  test_assert(segment_size(4, 4, 2) == 1);
  test_assert(segment_size(4, 4, 3) == 1);

  test_assert(segment_size(5, 4, 0) == 2);
  test_assert(segment_size(5, 4, 1) == 2);
  test_assert(segment_size(5, 4, 2) == 1);
  test_assert(segment_size(5, 4, 3) == 0);

  test_assert(segment_size(16, 5, 0) == 4);
  test_assert(segment_size(16, 5, 1) == 4);
  test_assert(segment_size(16, 5, 2) == 4);
  test_assert(segment_size(16, 5, 3) == 4);
  test_assert(segment_size(16, 5, 4) == 0);
}



// Test segment_size with normal algorithm

void
test_normal()
{
  using vsip::impl::segment_size;

  test_assert(segment_size(1, 4, 0) == 1);
  test_assert(segment_size(1, 4, 1) == 0);
  test_assert(segment_size(1, 4, 2) == 0);
  test_assert(segment_size(1, 4, 3) == 0);

  test_assert(segment_size(2, 4, 0) == 1);
  test_assert(segment_size(2, 4, 1) == 1);
  test_assert(segment_size(2, 4, 2) == 0);
  test_assert(segment_size(2, 4, 3) == 0);

  test_assert(segment_size(3, 4, 0) == 1);
  test_assert(segment_size(3, 4, 1) == 1);
  test_assert(segment_size(3, 4, 2) == 1);
  test_assert(segment_size(3, 4, 3) == 0);

  test_assert(segment_size(4, 3, 0) == 2);
  test_assert(segment_size(4, 3, 1) == 1);
  test_assert(segment_size(4, 3, 2) == 1);

  test_assert(segment_size(4, 4, 0) == 1);
  test_assert(segment_size(4, 4, 1) == 1);
  test_assert(segment_size(4, 4, 2) == 1);
  test_assert(segment_size(4, 4, 3) == 1);

  test_assert(segment_size(5, 4, 0) == 2);
  test_assert(segment_size(5, 4, 1) == 1);
  test_assert(segment_size(5, 4, 2) == 1);
  test_assert(segment_size(5, 4, 3) == 1);

  test_assert(segment_size(16, 5, 0) == 4);
  test_assert(segment_size(16, 5, 1) == 3);
  test_assert(segment_size(16, 5, 2) == 3);
  test_assert(segment_size(16, 5, 3) == 3);
  test_assert(segment_size(16, 5, 4) == 3);
}



int
main(int argc, char **argv)
{
  vsipl init(argc, argv);

#if VSIP_IMPL_USE_PAS_SEGMENT_SIZE
  test_pas();
#else
  test_normal();
#endif
}

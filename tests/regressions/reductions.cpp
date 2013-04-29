//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Regression tests for summation-based reductions on very large views.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/math.hpp>
#include <vsip/map.hpp>
#include <vsip/parallel.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>

using namespace vsip;

void
large_reduction_tests()
{
  using namespace vsip_csl;
  typedef float T;

  // This test is constructed to catch an error reported by a customer
  // in which a very large sum gave inaccurate results.  As it turns out
  // the 2^24 bits of mantissa stored in a 32-bit float mean that for
  // sufficiently large sums, the result will lose accuracy.  This has 
  // been partially addressed by a patch that uses a sub-blocking
  // strategy, storing partial results.   Even if we do a better
  // implelentation of this later (involving a recursive approach, likely)
  // we still want this test.
  T const base_value = T(0.1f);
  length_type const N = 1 << 24;
  Vector<float> vec(N, base_value);

  // Note: 'equal()' uses a relative error threshold of 10^-4.  The 
  // function 'almost_equal()' allows the error threshold to be specified.
  // At present, it is set at 2%.
  test_assert(almost_equal<T>(sumval(vec),    N * base_value,     2e-2));
  test_assert(almost_equal<T>(meanval(vec),   base_value,         2e-2));
  test_assert(almost_equal<T>(sumsqval(vec),  N * sq(base_value), 2e-2));
  test_assert(almost_equal<T>(meansqval(vec), sq(base_value),     2e-2));
}


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);
   
  large_reduction_tests();
}

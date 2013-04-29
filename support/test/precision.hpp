//
// Copyright (c) 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under theLicense.
// license contained in the accompanying LICENSE.GPL file.

#ifndef test_precision_hpp_
#define test_precision_hpp_

#include <vsip/support.hpp>

namespace test
{
template <typename T>
struct precision
{
  typedef T type;
  typedef typename scalar_of<T>::type scalar_type;

  static T eps;

  // Determine the lowest bit of precision.
  static void init()
  {
    eps = scalar_type(1);

    // Without 'volatile', ICC avoids rounding and computes precision of
    // long double for all types.
    volatile scalar_type a = 1.0 + eps;
    volatile scalar_type b = 1.0;

    while (a - b != scalar_type())
    {
      eps = 0.5 * eps;
      a = 1.0 + eps;
    }
  }
};

template <typename T>
T precision<T>::eps;

} // namespace test

#endif

/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/test-precision.cpp
    @author  Jules Bergmann
    @date    2005-09-12
    @brief   VSIPL++ CodeSourcery Library: Precision traits for tests.
*/

#ifndef VSIP_CSL_TEST_PRECISION_HPP
#define VSIP_CSL_TEST_PRECISION_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>



namespace vsip_csl
{

/***********************************************************************
  Declarations
***********************************************************************/

template <typename T>
struct Precision_traits
{
  typedef T type;
  typedef typename vsip::impl::Scalar_of<T>::type scalar_type;

  static T eps;

  // Determine the lowest bit of precision.

  static void compute_eps()
  {
    eps = scalar_type(1);

    // Without 'volatile', ICC avoid rounding and compute precision of
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

} // namespace vsip_csl

#endif // VSIP_CSL_TEST_PRECISION_HPP

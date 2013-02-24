/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef test_ramp_hpp_
#define test_ramp_hpp_

#include <vsip/vector.hpp>

template <typename T>
vsip::const_Vector<T>
test_ramp(T a, T b, vsip::length_type len)
{
  vsip::Vector<T> r(len);
  for (vsip::index_type i = 0; i < len; ++i)
    r.put(i, a + T(i)*b);
  return r;
}

template <typename T, typename B>
vsip::Vector<T, B>
test_ramp(vsip::Vector<T, B> view, T a, T b)
{
  for (vsip::index_type i=0; i<view.size(); ++i)
    view.put(i, a + T(i)*b);
  return view;
}

#endif

//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef test_ramp_hpp_
#define test_ramp_hpp_

#include <vsip/vector.hpp>

namespace test
{
template <typename T>
vsip::const_Vector<T>
ramp(T a, T b, vsip::length_type len)
{
  vsip::Vector<T> r(len);
  for (vsip::index_type i = 0; i < len; ++i)
    r.put(i, a + T(i)*b);
  return r;
}

template <typename T, typename B>
vsip::Vector<T, B>
ramp(vsip::Vector<T, B> view, T a, T b)
{
  for (vsip::index_type i=0; i<view.size(); ++i)
    view.put(i, a + T(i)*b);
  return view;
}
} // namespace test

#endif

/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/selgen/generation.cpp
    @author  Stefan Seefeld
    @date    2009-09-09
    @brief   VSIPL++ Library: Python bindings for generation types and functions.

*/
#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <vsip/selgen.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include "../block.hpp"
#include <memory>

namespace pyvsip
{
template <typename T>
vsip::Vector<T, Block<1, T> >
ramp(T a, T b, vsip::length_type len)
{
  vsip::Vector<T, Block<1, T> > result(len);
  result = vsip::ramp(a, b, len);
  return result;
}

template <typename T>
vsip::Vector<T, Block<1, T> >
clip1(vsip::Vector<T, Block<1, T> > v, T lt, T ut, T lc, T uc)
{
  vsip::Vector<T, Block<1, T> > result(v.length());
  result = vsip::clip(v, lt, ut, lc, uc);
  return result;
}

template <typename T>
vsip::Matrix<T, Block<2, T> >
clip2(vsip::Matrix<T, Block<2, T> > m, T lt, T ut, T lc, T uc)
{
  vsip::Matrix<T, Block<2, T> > result(m.size(0), m.size(1));
  result = vsip::clip(m, lt, ut, lc, uc);
  return result;
}

template <typename T>
vsip::Vector<T, Block<1, T> >
invclip1(vsip::Vector<T, Block<1, T> > v, T lt, T mt, T ut, T lc, T uc)
{
  vsip::Vector<T, Block<1, T> > result(v.length());
  result = vsip::invclip(v, lt, mt, ut, lc, uc);
  return result;
}

template <typename T>
vsip::Matrix<T, Block<2, T> >
invclip2(vsip::Matrix<T, Block<2, T> > m, T lt, T mt, T ut, T lc, T uc)
{
  vsip::Matrix<T, Block<2, T> > result(m.size(0), m.size(1));
  result = vsip::invclip(m, lt, mt, ut, lc, uc);
  return result;
}

template <typename T>
void define_generation()
{
  bpl::def("ramp", ramp<T>);
  bpl::def("clip", clip1<T>);
  bpl::def("clip", clip2<T>);
  bpl::def("invclip", invclip1<T>);
  bpl::def("invclip", invclip2<T>);
}

}

BOOST_PYTHON_MODULE(generation)
{
  using namespace pyvsip;
  //define_generation<float>();
  define_generation<double>();
}

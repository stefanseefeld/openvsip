/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/signal/freqswap.cpp
    @author  Stefan Seefeld
    @date    2009-09-11
    @brief   VSIPL++ Library: Python bindings for signal module.

*/
#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <vsip/signal.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include "../block.hpp"

namespace pyvsip
{

template <typename T>
vsip::Vector<T, Block<1, T> >
freqswap1(vsip::Vector<T, Block<1, T> > in) { return vsip::freqswap(in);}

template <typename T>
vsip::Matrix<T, Block<2, T> >
freqswap2(vsip::Matrix<T, Block<2, T> > in)
{ return vsip::freqswap(in);}
}

BOOST_PYTHON_MODULE(freqswap)
{
  using namespace pyvsip;

  bpl::def("freqswap", freqswap1<float>);
  bpl::def("freqswap", freqswap1<double>);
  bpl::def("freqswap", freqswap2<float>);
  bpl::def("freqswap", freqswap2<double>);
}

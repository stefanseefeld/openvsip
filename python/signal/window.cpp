/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/signal/window.cpp
    @author  Stefan Seefeld
    @date    2009-09-20
    @brief   VSIPL++ Library: Python bindings for signal module.

*/
#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <vsip/signal.hpp>
#include <vsip/vector.hpp>
#include "../block.hpp"

namespace pyvsip
{
typedef vsip::Vector<float, Block<1, float> > vector_type;

vector_type blackman(vsip::length_type len)
{ return vsip::blackman(len);}
vector_type cheby(vsip::length_type len, float ripple)
{ return vsip::cheby(len, ripple);}
vector_type hanning(vsip::length_type len)
{ return vsip::hanning(len);}
vector_type kaiser(vsip::length_type len, float beta)
{ return vsip::kaiser(len, beta);}
  
}

BOOST_PYTHON_MODULE(window)
{
  using namespace pyvsip;
  namespace bpl = boost::python;

  bpl::def("blackman", blackman);
  bpl::def("cheby", cheby);
  bpl::def("hanning", hanning);
  bpl::def("kaiser", kaiser);
}

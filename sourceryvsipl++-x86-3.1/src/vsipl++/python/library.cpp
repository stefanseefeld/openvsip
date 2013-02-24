/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/library.cpp
    @author  Stefan Seefeld
    @date    2009-08-12
    @brief   VSIPL++ Library: Python bindings for library initialization.

*/
#include <boost/python.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>

namespace bpl = boost::python;

BOOST_PYTHON_MODULE(library)
{
  bpl::class_<vsip::vsipl, boost::noncopyable> vsipl("library");
}

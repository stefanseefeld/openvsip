//
// Copyright (c) 2015 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/python/block.hpp>
#include <boost/python.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <stdexcept>
#include <cstdarg>
#include <iostream>

namespace bpl = boost::python;
using namespace vsip;

namespace
{
void convert(std::runtime_error const &e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

}

BOOST_PYTHON_MODULE(support)
{
  bpl::register_exception_translator<std::runtime_error>(convert);
  bpl::class_<Domain<1> > dom("domain", bpl::init<length_type>());
  dom.def(bpl::init<index_type, stride_type, length_type>());
  dom.add_property("first", &Domain<1>::first);
  dom.add_property("stride", &Domain<1>::stride);
  dom.add_property("length", &Domain<1>::length);

  bpl::implicitly_convertible<int, Domain<1> >();
}

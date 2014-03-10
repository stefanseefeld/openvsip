//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/python/block.hpp>
#include <boost/python.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <stdexcept>

namespace bpl = boost::python;

namespace
{
void convert(std::runtime_error const &e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}

}

BOOST_PYTHON_MODULE(library)
{
  bpl::class_<vsip::vsipl, boost::noncopyable> vsipl("library");
  bpl::register_exception_translator<std::runtime_error>(convert);
}

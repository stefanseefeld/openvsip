//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <boost/python.hpp>
#include <vsip/domain.hpp>
#include <stdexcept>
#include <cstdarg>

namespace bpl = boost::python;

namespace
{
struct converter
{
  static PyObject *convert(vsip::Index<1> const &i)
  {
    return bpl::incref(bpl::object(i[0]).ptr());
  }
  static PyObject *convert(vsip::Index<2> const &i)
  {
    return bpl::incref(bpl::make_tuple(i[0], i[1]).ptr());
  }
  static PyObject *convert(vsip::Index<3> const &i)
  {
    return bpl::incref(bpl::make_tuple(i[0], i[1], i[2]).ptr());
  }
};

}

BOOST_PYTHON_MODULE(types)
{
  bpl::to_python_converter<vsip::Index<1>, converter>();
  bpl::to_python_converter<vsip::Index<2>, converter>();
  bpl::to_python_converter<vsip::Index<3>, converter>();
}

//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/python/block.hpp>
#include <ovxx/view.hpp>
#include <ovxx/assign.hpp>
#include <ovxx/domain_utils.hpp>

namespace bpl = boost::python;

namespace
{

template <typename T, vsip::dimension_type D, typename T1>
bpl::object cast(ovxx::python::Block<D, T1> const &b)
{
  vsip::Domain<D> dom = ovxx::block_domain<D>(b);
  boost::shared_ptr<ovxx::python::Block<D, T> > other(new ovxx::python::Block<D, T>(dom));
  ovxx::assign<D>(*other, b);
  return bpl::object(other);
}

template <vsip::dimension_type D, typename T>
bpl::object as_type(bpl::object type, ovxx::python::Block<D, T> const &b)
{
  if (PyType_Check(type.ptr()))
  {
    if (type.ptr() == (PyObject*)&PyInt_Type)
      return cast<long>(b);
    else if (type.ptr() == (PyObject*)&PyFloat_Type)
      return cast<double>(b);
    else if (type.ptr() == (PyObject*)&PyComplex_Type)
      return cast<vsip::complex<double> >(b);
    else if (type.ptr() == (PyObject*)&PyFloatArrType_Type)
      return cast<float>(b);
    else if (type.ptr() == (PyObject*)&PyCFloatArrType_Type)
      return cast<vsip::complex<float> >(b);
    else if (type.ptr() == (PyObject*)&PyFloat64ArrType_Type)
      return cast<double>(b);
    else throw std::runtime_error("unsupported dtype");
  }
  else throw std::runtime_error("argument not a dtype");
}

}


BOOST_PYTHON_MODULE(_cast)
{
  bpl::def("as_type", as_type<1, int>);
  bpl::def("as_type", as_type<2, int>);
  bpl::def("as_type", as_type<1, float>);
  bpl::def("as_type", as_type<2, float>);
  bpl::def("as_type", as_type<1, double>);
  bpl::def("as_type", as_type<2, double>);
  // bpl::def("as_type", as_type<1, vsip::complex<float> >);
  // bpl::def("as_type", as_type<2, vsip::complex<float> >);
  // bpl::def("as_type", as_type<1, vsip::complex<double> >);
  // bpl::def("as_type", as_type<2, vsip::complex<double> >);
}

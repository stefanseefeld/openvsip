/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/signal/histogram.cpp
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
vsip::Vector<int, Block<1, int> >
histogram1(vsip::Histogram<vsip::const_Vector, T> h,
	   vsip::Vector<T, Block<1, T> > data, bool accumulate = false)
{ return h(data, accumulate);}

template <typename T>
vsip::Vector<int, Block<1, int> >
histogram2(vsip::Histogram<vsip::const_Vector, T> h,
	   vsip::Matrix<T, Block<2, T> > data, bool accumulate = false)
{ return h(data, accumulate);}

template <typename T>
void define_histogram(char const *type_name)
{
  typedef vsip::Histogram<vsip::const_Vector, T> histo_type;

  bpl::class_<histo_type> histo(type_name,
				bpl::init<T, T, int>());
  histo.def(bpl::init<T, T, vsip::Vector<int, Block<1, int> > >());
  histo.def("__call__", histogram1<T>, bpl::arg("accumulate") = false);
  histo.def("__call__", histogram2<T>, bpl::arg("accumulate") = false);
}

template <typename T>
bpl::object create_histogram(T l, T u, vsip::length_type b)
{
  return bpl::object(vsip::Histogram<vsip::const_Vector, T>(l, u, b));
}

bpl::object 
create_histogram_from_type(bpl::object type,
			   bpl::object l, bpl::object u, vsip::length_type b)
{
  if (PyType_Check(type.ptr()))
  {
    if (type.ptr() == (PyObject*)&PyInt_Type)
      return create_histogram<long>(bpl::extract<long>(l),
				    bpl::extract<long>(u),
				    b);
    else if (type.ptr() == (PyObject*)&PyFloat_Type)
      return create_histogram<float>(bpl::extract<float>(l),
				     bpl::extract<float>(u),
				     b);
    else throw std::runtime_error("unsupported type");
  }
  else throw std::runtime_error("argument not a type");
}
}

BOOST_PYTHON_MODULE(histogram)
{
  using namespace pyvsip;

  define_histogram<float>("FHistogram");
  define_histogram<double>("DHistogram");
  bpl::def("histogram", create_histogram_from_type);
}

//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <boost/python.hpp>
#include <ovxx/python.hpp>
#include <vsip/initfin.hpp>
#include <vsip/selgen.hpp>

// Use an (embedded) Python script as prototype, then compute the
// same in C++, and validate the result by comparing with the Python version.
//
// This demonstrates a hybrid development workflow wherein algorithms are
// prototyped with Python VSIP, then maintained as "Gold Standards".
// The code can be easily transcribed into VSIPL++, and both versions
// continuously validated to make sure the implementation is correct and 
// neither falls out of sync with the other.

namespace bpl = boost::python;

// Define a simple validator function
// that performs an elementwise comparison of
// a VSIPL++ vector with a Python VSIP vector.
template <typename T>
bool validate(vsip::Vector<T> cxx, bpl::object py)
{
  bpl::object main = bpl::import("__main__");
  bpl::object global(main.attr("__dict__"));
  bpl::exec("from numpy import array\n", global);
  global["cxx"] = cxx;
  global["py"] = py;
  bpl::object result = bpl::eval("array(cxx == py).all()", global);
  return result;
}


// Define a prototype algorithm using Python VSIP 
// and return a result vector as reference.
bpl::object prototype()
{
  bpl::object main = bpl::import("__main__");
  bpl::object global(main.attr("__dict__"));
  bpl::exec("from vsip import float32 \n"
	    "from vsip import selgen \n"
	    "v = selgen.ramp(1, 2, 16, dtype=float32)\n",
	    global);
  return global["v"];
}

// Implement the prototype algorithm using VSIPL++
vsip::Vector<float> implementation()
{
  return vsip::ramp<float>(1, 2, 16);
}

int main(int argc, char **argv)
{
  using namespace ovxx;

  Py_Initialize();
  python::initialize();
  python::load_converter<float>();

  vsipl library(argc, argv);

  try
  {
    bpl::object p = prototype();
    vsip::Vector<float> cxx = implementation();
    if (validate(cxx, p))
      std::cout << "PASS (prototype and implementation yield the same result)" << std::endl;
    else
      std::cout << "FAIL (prototype and implementation yirld different results)" << std::endl;

  }
  catch (...)
  {
    PyErr_Print();
  }
}

//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <boost/python.hpp>
#include <ovxx/python.hpp>
#include <vsip/initfin.hpp>
#include <vsip/signal.hpp>

// Define a VSIPL++ vector, then use Python VSIP to plot it.
//
// This demonstrates how to embed a Python interpreter into
// a C++ application and share VSIP objects across the language
// boundary.

int main(int argc, char **argv)
{
  namespace bpl = boost::python;
  using namespace ovxx;

  Py_Initialize();
  python::initialize();
  python::load_converter<float>();

  vsipl library(argc, argv);

  Vector<float> filter = blackman(1024);

  bpl::object main = bpl::import("__main__");
  bpl::object global(main.attr("__dict__"));
  try
  {
    global["vsip"] = bpl::import("vsip");
    global["filter"] = filter;
    bpl::object result = 
      bpl::exec("from matplotlib.pyplot import *\n"
		"plot(filter)\n"
		"show()\n",
		global);
  }
  catch (...)
  {
    PyErr_Print();
  }
}

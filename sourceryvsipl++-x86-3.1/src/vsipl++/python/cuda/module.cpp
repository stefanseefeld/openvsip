/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip_csl/cuda.hpp>
#include "block.hpp"
#include <memory>

using namespace vsip_csl::cuda;

namespace pyvsip
{

void function_param_setv(Function &f, int offset, bpl::object buffer)
{ 
  void const *buf;
  Py_ssize_t length;
  if (PyObject_AsReadBuffer(buffer.ptr(), &buf, &length))
    throw bpl::error_already_set();
  f.param_setv(offset, const_cast<void *>(buf), length);
}


std::auto_ptr<Module> 
from_buffer(bpl::object buffer)
{
  char const *mod_buf;
  Py_ssize_t lenth;
  if (PyObject_AsCharBuffer(buffer.ptr(), &mod_buf, &lenth))
    throw bpl::error_already_set();
  return std::auto_ptr<Module>(new Module((void const*)mod_buf));
}

}

BOOST_PYTHON_MODULE(module)
{  
  using namespace pyvsip;
  
  bpl::class_<Function> function("Function", bpl::no_init);
  function.def("set_block_shape", &Function::set_block_shape);
  function.def("set_shared_size", &Function::set_shared_size);
  function.def("param_setv", function_param_setv);
  function.def("param_set_size", &Function::param_set_size);
  function.def("launch", &Function::launch);
  function.def("launch_grid", &Function::launch_grid);

  bpl::class_<Module, boost::noncopyable> module("Module", bpl::init<char const *>());
  module.def("__init__", bpl::make_constructor(from_buffer));
  module.def("get_function", &Module::get_function);
  module.def("get_global", &Module::get_global);
}

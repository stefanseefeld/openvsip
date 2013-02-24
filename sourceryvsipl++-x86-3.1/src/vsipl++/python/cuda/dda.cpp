/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <boost/python.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip_csl/cuda.hpp>
#include <vsip/opt/cuda/dda.hpp>
#include "block.hpp"
#include <memory>

using namespace vsip::impl::cuda;

namespace pyvsip
{

template <typename B>
std::auto_ptr<dda::Data<B, vsip::dda::inout> >
create_data_from_vector(vsip::Vector<typename B::value_type, B> v)
{
  typedef dda::Data<B, vsip::dda::inout> data_type;
  return std::auto_ptr<data_type>(new data_type(v.block()));
}

template <typename B>
std::auto_ptr<dda::Data<B, vsip::dda::inout> >
create_data_from_matrix(vsip::Matrix<typename B::value_type, B> v)
{
  typedef dda::Data<B, vsip::dda::inout> data_type;
  return std::auto_ptr<data_type>(new data_type(v.block()));
}

template <typename B>
CUdeviceptr
data_ptr(dda::Data<B, vsip::dda::inout> &d)
{
  return (CUdeviceptr)(size_t)d.ptr();
}
template <typename B>
PyObject *
data_ptr_to_long(dda::Data<B, vsip::dda::inout> &d)
{
  return PyLong_FromUnsignedLong(data_ptr(d));
}

template <typename T>
void define_vector_data(char const *name)
{
  typedef Block<1, T> B;
  bpl::class_<vsip::impl::cuda::dda::Data<B, vsip::dda::inout>, boost::noncopyable>
    data(name, bpl::no_init);
  data.def("__init__", bpl::make_constructor(create_data_from_vector<B>));
  data.def("__int__", data_ptr<B>);
  data.def("__index__", data_ptr<B>);
  data.def("__long__", data_ptr_to_long<B>);
}

template <typename T>
void define_matrix_data(char const *name)
{
  typedef Block<2, T> B;
  bpl::class_<vsip::impl::cuda::dda::Data<B, vsip::dda::inout>, boost::noncopyable>
    data(name, bpl::no_init);
  data.def("__init__", bpl::make_constructor(create_data_from_matrix<B>));
  data.def("__int__", data_ptr<B>);
  data.def("__index__", data_ptr<B>);
  data.def("__long__", data_ptr_to_long<B>);
}

}

BOOST_PYTHON_MODULE(dda)
{  
  using namespace pyvsip;

  define_vector_data<float>("FVData");
  define_vector_data<std::complex<float> >("CVData");

  define_matrix_data<float>("FMData");
  define_matrix_data<std::complex<float> >("CMData");
}

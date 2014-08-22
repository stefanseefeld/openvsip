//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_python_converter_hpp_
#define ovxx_python_converter_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <iostream>

namespace ovxx
{
namespace python
{
namespace detail
{
template <typename T> struct block_mod;
#define OVXX_PYTHON_IMPORT(T, M)       	   \
template <> struct block_mod<T>            \
{					   \
  static void import() { bpl::import(#M);} \
};

OVXX_PYTHON_IMPORT(bool, vsip.iblock)
OVXX_PYTHON_IMPORT(int, vsip.bblock)
OVXX_PYTHON_IMPORT(float, vsip.fblock)
OVXX_PYTHON_IMPORT(double, vsip.dblock)
OVXX_PYTHON_IMPORT(complex<float>, vsip.cfblock)
OVXX_PYTHON_IMPORT(complex<double>, vsip.cdblock)

}

template <typename T>
struct vector_to_python
{
  static PyObject *convert(vsip::Vector<T> v)
  {
    detail::block_mod<T>::import();
    // We can't keep a reference to arbitrary C++ blocks,
    // so we pass by-value.
    shared_ptr<Block<1, T> > block(new Block<1, T>(v.size()));
    assign<1>(*block, v.block());
    bpl::dict ns;
    ns["vsip"] = bpl::import("vsip");
    ns["tmp"] = bpl::object(block);
    bpl::object pv = bpl::eval("vsip.vector(block=tmp)", ns);
    return bpl::incref(pv.ptr());
  }
};
template <typename T>
struct matrix_to_python
{
  static PyObject *convert(vsip::const_Matrix<T> m)
  {
    detail::block_mod<T>::import();
    // We can't keep a reference to arbitrary C++ blocks,
    // so we pass by-value.
    shared_ptr<Block<2, T> > block(new Block<2, T>(Domain<2>(m.size(0), m.size(1))));
    assign<2>(*block, m.block());
    bpl::dict ns;
    ns["vsip"] = bpl::import("vsip");
    ns["tmp"] = bpl::object(block);
    bpl::object pm = bpl::eval("vsip.matrix(block=tmp)", ns);
    return bpl::incref(pm.ptr());
  }
};

template <typename T>
void load_converter()
{
  bpl::to_python_converter<vsip::Vector<T>, vector_to_python<T> >();
  bpl::to_python_converter<vsip::const_Matrix<T>, matrix_to_python<T> >();
}

} // namespace ovxx::python
} // namespace ovxx

#endif

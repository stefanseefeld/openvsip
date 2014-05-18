//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef block_api_hpp_
#define block_api_hpp_

#include <ovxx/python/block.hpp>
//#include "converter.hpp"
#include <ovxx/view/fns_elementwise.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <iostream>

namespace pyvsip
{
using namespace ovxx;
using namespace ovxx::python;

template <typename T>
bpl::object get_dtype()
{
  return bpl::object(bpl::detail::new_reference
     (reinterpret_cast<PyObject*>(PyArray_DescrFromType(get_typenum<T>()))));
}

template <dimension_type D, typename T>
length_type total_size(Block<D, T> const &b) { return b.size();}
template <dimension_type D, typename T>
length_type size(Block<D, T> const &b, dimension_type dim, dimension_type d)
{ return b.size(dim, d);}
template <dimension_type D, typename T>
bpl::tuple shape(Block<D, T> const &b)
{
  bpl::list l;
  for (dimension_type d = 0; d != D; ++d)
    l.append(b.size(D, d));
  return bpl::make_tuple(l);
}

template <dimension_type D, typename T> struct traits;
template <typename T> 
struct traits<1,T>
{
  typedef Block<1,T> B;
  static boost::shared_ptr<B> construct(length_type l)
  { return boost::shared_ptr<B>(new B(l));}
  static T get(B const &b, index_type i) 
  {
    OVXX_PRECONDITION(i < b.size(1, 0));
    return b.get(i);
  }
  static void put(B &b, index_type i, T v)
  {
    OVXX_PRECONDITION(i < b.size(1, 0));
    b.put(i, v);
  }
};
template <typename T> 
struct traits<2,T>
{
  typedef Block<2,T> B;
  static boost::shared_ptr<B> construct(length_type r, length_type c)
  { return boost::shared_ptr<B>(new Block<2,T>(Domain<2>(r,c)));}
  static T get(B const &b, index_type i, index_type j)
  {
    OVXX_PRECONDITION(i < b.size(2, 0));
    OVXX_PRECONDITION(j < b.size(2, 1));
    return b.get(i, j);
  }
  static void put(B &b, index_type i, index_type j, T v)
  {
    OVXX_PRECONDITION(i < b.size(2, 0));
    OVXX_PRECONDITION(j < b.size(2, 1));
    b.put(i, j, v);
  }
};
template <typename T> 
struct traits<3,T>
{
  typedef Block<3,T> B;
  static boost::shared_ptr<B> construct(length_type d0, length_type d1, length_type d2)
  { return boost::shared_ptr<B>(new Block<3,T>(Domain<3>(d0,d1,d2)));}
  static T get(B const &b, index_type i, index_type j, index_type k)
  {
    OVXX_PRECONDITION(i < b.size(3, 0));
    OVXX_PRECONDITION(j < b.size(3, 1));
    OVXX_PRECONDITION(k < b.size(3, 2));
    return b.get(i, j, k);
  }
  static void put(B &b, index_type i, index_type j, index_type k, T v)
  {
    OVXX_PRECONDITION(i < b.size(3, 0));
    OVXX_PRECONDITION(j < b.size(3, 1));
    OVXX_PRECONDITION(k < b.size(3, 2));
    b.put(i, j, k, v);
  }
};

// The one-argument constructor needs some manual dispatching,
// as the argument may be an array or a length.
template <typename T>
bpl::object construct(bpl::object o)
{
  if (!PyArray_Check(o.ptr()))
  {
    bpl::extract<length_type> e(o);
    if (e.check())
      return bpl::object(boost::shared_ptr<Block<1, T> >(new Block<1, T>(e())));
    else
      PYVSIP_THROW(TypeError, "argument is neither a numpy array nor a length");
  }
  PyArrayObject *a = reinterpret_cast<PyArrayObject*>(o.ptr());
  if (!is_storage_compatible<T>(a))
    PYVSIP_THROW(TypeError, "argument is numpy array of wrong type");
  int dim = PyArray_NDIM(a);
  npy_intp *dims = PyArray_SHAPE(a);
  switch (dim)
  {
    case 1:
      return bpl::object(boost::shared_ptr<Block<1, T> >(new Block<1, T>(o)));
    case 2:
      return bpl::object(boost::shared_ptr<Block<2, T> >(new Block<2, T>(o)));
    case 3:
      return bpl::object(boost::shared_ptr<Block<3, T> >(new Block<3, T>(o)));
    default:
      PYVSIP_THROW(ValueError, "unsupported shape");
  }
}

// Calculate a domain corresponding to the given slice, assuming the given length.
// The stride parameter is needed to account for non-unit-stride access (i.e. when
// slicing subblocks)
inline Domain<1> slice_to_domain(bpl::slice s, length_type size, stride_type stride)
{
  Py_ssize_t start, stop, step, length;
  int status = PySlice_GetIndicesEx((PySliceObject*)s.ptr(), size,
				    &start, &stop, &step, &length);
  return Domain<1>(start*stride, step*stride, length);
}

template <typename T>
bpl::object subblock1(Block<1, T> &b, bpl::slice s)
{
  Domain<1> dom = slice_to_domain(s, b.size(1, 0), b.stride(1, 0));
  return bpl::object(boost::shared_ptr<Block<1, T> >(new Block<1, T>(b, dom.first(), dom)));
}

template <typename T>
bpl::object subblock2(Block<2, T> &b, bpl::slice i, bpl::slice j)
{
  Domain<1> domi = slice_to_domain(i, b.size(2, 0), b.stride(2, 0));
  Domain<1> domj = slice_to_domain(j, b.size(2, 1), b.stride(2, 1));
  int offset = domi.first() + domj.first();
  return bpl::object(boost::shared_ptr<Block<2, T> >(new Block<2, T>
    (b, offset, Domain<2>(domi, domj))));
}

template <typename T>
bpl::object get_row(Block<2, T> &b, int i)
{
  // FIXME: this assumes row-major ordering
  if (i < 0) i += b.size(2, 0);
  int start = i * b.stride(2, 0);
  Domain<1> domain(0, 1, b.size(2, 1));
  return bpl::object(boost::shared_ptr<Block<1, T> >(new Block<1, T>(b, start, domain)));
}

template <typename T>
bpl::object get_col(Block<2, T> &b, int i)
{
  // FIXME: this assumes row-major ordering
  if (i < 0) i += b.size(2, 1);
  int start = i * b.stride(2, 1);
  Domain<1> domain(start, b.stride(2, 0), b.size(2, 0));
  return bpl::object(boost::shared_ptr<Block<1, T> >(new Block<1, T>(b, start, domain)));
}

template <dimension_type D, typename T>
bpl::object real(Block<D, complex<T> > &b)
{
  return bpl::object(boost::shared_ptr<Block<D, T> >
    (new Block<D, T>(b, 0, ovxx::block_domain<D>(b)*2)));
}

template <dimension_type D, typename T>
bpl::object imag(Block<D, complex<T> > &b)
{
  return bpl::object(boost::shared_ptr<Block<D, T> >
    (new Block<D, T>(b, 1, ovxx::block_domain<D>(b)*2)));
}

template <dimension_type D, typename T>
void assign(Block<D, T> &b, Block<D, T> const &other)
{
  // TODO: dispatch to a dense assignment, if possible
  ovxx::assign<D>(b, other);
}

template <dimension_type D, typename T>
void assign_scalar(Block<D, T> &b, T val)
{
  // TODO: dispatch to a dense assignment, if possible
  ovxx::expr::Scalar<D, T> scalar(val);
  ovxx::assign<D>(b, scalar);
}

template <dimension_type D, typename T>
bpl::object eq(Block<D, T> const &b1, Block<D, T> const &b2)
{
  typedef Block<D, T> B;
  typename ovxx::view_of<B>::const_type v1(const_cast<B&>(b1));
  typename ovxx::view_of<B>::const_type v2(const_cast<B&>(b2));
  Domain<D> dom = ovxx::block_domain<D>(b1);
  Block<D, bool> *result = new Block<D, bool>(dom);
  typename ovxx::view_of<Block<D, bool> >::type r(*result);
  r = vsip::eq(v1, v2);
  return bpl::object(boost::shared_ptr<Block<D, bool> >(result));
}

template <dimension_type D, typename T>
Block<D, T> &iadd(Block<D, T> &b1, Block<D, T> const &b2)
{
  typedef Block<D, T> B;
  typename ovxx::view_of<B>::type v1(b1);
  typename ovxx::view_of<B>::const_type v2(const_cast<B&>(b2));
  v1 += v2;
  return b1;
}

template <dimension_type D, typename T>
Block<D, T> &isub(Block<D, T> &b1, Block<D, T> const &b2)
{
  typedef Block<D, T> B;
  typename ovxx::view_of<B>::type v1(b1);
  typename ovxx::view_of<B>::const_type v2(const_cast<B&>(b2));
  v1 -= v2;
  return b1;
}

template <dimension_type D, typename T>
Block<D, T> &imul(Block<D, T> &b1, Block<D, T> const &b2)
{
  typedef Block<D, T> B;
  typename ovxx::view_of<B>::type v1(b1);
  typename ovxx::view_of<B>::const_type v2(const_cast<B&>(b2));
  v1 *= v2;
  return b1;
}

template <dimension_type D, typename T>
Block<D, T> &idiv(Block<D, T> &b1, Block<D, T> const &b2)
{
  typedef Block<D, T> B;
  typename ovxx::view_of<B>::type v1(b1);
  typename ovxx::view_of<B>::const_type v2(const_cast<B&>(b2));
  v1 /= v2;
  return b1;
}

template <typename C, typename T>
void define_compound_assignment(C &block, T)
{
  block.def(bpl::self += bpl::self);
  block.def(bpl::self += T());
  block.def(bpl::self -= bpl::self);
  block.def(bpl::self -= T());
  block.def(bpl::self *= bpl::self);
  block.def(bpl::self *= T());
  block.def(bpl::self /= bpl::self);
  block.def(bpl::self /= T());
}

template <dimension_type D, typename C, typename T>
void define_complex_subblocks(C &block, T) {}

template <dimension_type D, typename C, typename T>
void define_complex_subblocks(C &block, complex<T>)
{
  block.def("real", real<D, T>);
  block.def("imag", imag<D, T>);
}

template <dimension_type D, typename T>
void define_block(char const *type_name)
{
  typedef Block<D, T> block_type;

  bpl::class_<block_type, boost::shared_ptr<block_type>, boost::noncopyable> 
    block(type_name, bpl::no_init);
  block.setattr("dtype", get_dtype<T>());
  /// Conversion to array.
  block.def("__array__", &block_type::array);
  block.def("assign", assign<D, T>);
  block.def("assign", assign_scalar<D, T>);

  block.def("copy", &block_type::copy);

  block.def("size", total_size<D,T>);
  block.def("size", size<D,T>);
  block.add_property("shape", shape<D,T>);
  block.def("get", &traits<D,T>::get);
  block.def("put", &traits<D,T>::put);

  block.def("__eq__", eq<D,T>);

  define_compound_assignment(block, T());
  define_complex_subblocks<D>(block, T());

  /// Construction from shape
  if (D != 1) // The case D==1 is covered in construct<T> below
    bpl::def("block", traits<D, T>::construct);
  /// Conversion from array.
  bpl::def("block", construct<T>);
  /// Construct subblock
  bpl::def("subblock", subblock1<T>);
  if (D == 2)
  {
    bpl::def("subblock", subblock2<T>);
    block.def("row", get_row<T>);
    block.def("col", get_col<T>);
  }

  // bpl::to_python_converter<vsip::const_Vector<T>, view_to_python<T> >();
  // bpl::to_python_converter<vsip::const_Matrix<T>, view_to_python<T> >();
}

}


#endif

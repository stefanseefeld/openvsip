//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef block_api_hpp_
#define block_api_hpp_

#include <ovxx/python/block.hpp>
#include <ovxx/view/fns_elementwise.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/parallel.hpp>
#include <iostream>
#include <ovxx/output.hpp>

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

template <dimension_type D, typename T, typename M>
length_type total_size(Block<D, T, M> const &b) { return b.size();}
template <dimension_type D, typename T, typename M>
length_type size(Block<D, T, M> const &b, dimension_type dim, dimension_type d)
{ return b.size(dim, d);}
template <dimension_type D, typename T, typename M>
bpl::tuple shape(Block<D, T, M> const &b)
{
  bpl::list l;
  for (dimension_type d = 0; d != D; ++d)
    l.append(b.size(D, d));
  return bpl::tuple(l);
}

template <dimension_type D, typename T, typename M = Local_map> struct traits;
template <typename T>
struct traits<1,T>
{
  typedef Block<1,T> B;
  static std::shared_ptr<B> construct(length_type l)
  { return std::shared_ptr<B>(new B(l));}
  static std::shared_ptr<B> construct_init(length_type l, T value)
  { return std::shared_ptr<B>(new B(l, value));}
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
template <typename T, typename M>
struct traits<1,T,M>
{
  typedef Block<1,T,M> B;
  static std::shared_ptr<B> construct(length_type l, M const &m)
  { return std::shared_ptr<B>(new B(l, m));}
  static std::shared_ptr<B> construct_init(length_type l, T value, M const &m)
  { return std::shared_ptr<B>(new B(l, value, m));}
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
  static std::shared_ptr<B> construct(length_type r, length_type c)
  { return std::shared_ptr<B>(new B(Domain<2>(r,c)));}
  static std::shared_ptr<B> construct_init(length_type r, length_type c, T value)
  { return std::shared_ptr<B>(new B(Domain<2>(r,c), value));}
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
template <typename T, typename M>
struct traits<2,T,M>
{
  typedef Block<2,T,M> B;
  static std::shared_ptr<B> construct(length_type r, length_type c, M const &m)
  { return std::shared_ptr<B>(new B(Domain<2>(r,c), m));}
  static std::shared_ptr<B> construct_init(length_type r, length_type c, T value, M const &m)
  { return std::shared_ptr<B>(new B(Domain<2>(r,c), value, m));}
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
  static std::shared_ptr<B> construct(length_type d0, length_type d1, length_type d2)
  { return std::shared_ptr<B>(new B(Domain<3>(d0,d1,d2)));}
  static std::shared_ptr<B> construct(length_type d0, length_type d1, length_type d2, T value)
  { return std::shared_ptr<B>(new B(Domain<3>(d0,d1,d2), value));}
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
template <typename T, typename M>
struct traits<3,T,M>
{
  typedef Block<3,T,M> B;
  static std::shared_ptr<B> construct(length_type d0, length_type d1, length_type d2, M const &m)
  { return std::shared_ptr<B>(new B(Domain<3>(d0,d1,d2), m));}
  static std::shared_ptr<B> construct(length_type d0, length_type d1, length_type d2, T value, M const &m)
  { return std::shared_ptr<B>(new B(Domain<3>(d0,d1,d2), value, m));}
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

template <dimension_type D, typename T>
bpl::handle<PyArrayObject> make_array_(Block<D, T> &);

// The one-argument constructor needs some manual dispatching,
// as the argument may be an array or a length.
template <typename T>
bpl::object construct(bpl::object o)
{
  if (!PyArray_Check(o.ptr()))
  {
    bpl::extract<length_type> e(o);
    if (e.check())
      return bpl::object(std::shared_ptr<Block<1, T> >(new Block<1, T>(e())));
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
    {
      std::shared_ptr<Block<1, T> > b(new Block<1, T>(dims[0]));
      bpl::handle<PyArrayObject> ba = make_array_(*b);
      PyArray_CopyInto(ba.get(), a);
      return bpl::object(b);
    }
    case 2:
    {
      Domain<2> dom(dims[0], dims[1]);
      std::shared_ptr<Block<2, T> > b(new Block<2, T>(dom));
      bpl::handle<PyArrayObject> ba = make_array_(*b);
      PyArray_CopyInto(ba.get(), a);
      return bpl::object(b);
    }
    default:
      PYVSIP_THROW(ValueError, "unsupported shape");
  }
}

// Arrays may refer back to a block, in which case
// the block needs to pin its (host) storage till 
// the end of the array's lifetime.
template <typename B>
struct array_backref
{
  array_backref(B &b) : b_(b) { b_.ref_ptr();}
  array_backref(array_backref const &r) : b_(r.b_) { b_.ref_ptr();}
  ~array_backref() { b_.unref_ptr();}
  B &b_;
};

// Construct an array referencing a block's data.
// The 'inner' function. Use with care.
template <dimension_type D, typename T>
bpl::handle<PyArrayObject> make_array_(Block<D, T> &block)
{
  npy_intp dims[D];
  npy_intp strides[D];
  for (dimension_type d = 0; d != D; ++d)
  {
    dims[d] = block.size(D, d);
    strides[d] = block.stride(D, d) * sizeof(T);
  }
  int flags = NPY_ARRAY_WRITEABLE;
  bpl::handle<PyArrayObject> 
    a((PyArrayObject *)
      PyArray_New(&PyArray_Type, D, dims, get_typenum<T>(),
		  strides, block.ptr(), 0, flags, 0));
  return a;
}


// Construct an array referencing a block's data.
// The 'outer' function. Add a backreference from the
// array to the block to make sure the shared data (storage)
// remains valid over the array's lifetime.
template <dimension_type D, typename T>
bpl::object make_array(bpl::object o)
{
  typedef Block<D, T> block_type;
  block_type &block = bpl::extract<block_type &>(o);
  bpl::handle<PyArrayObject> a = make_array_(block);
  std::shared_ptr<array_backref<block_type> > lock
    (new array_backref<block_type>(block));
  bpl::object backref(lock);
  PyArray_SetBaseObject(a.get(), bpl::incref(backref.ptr()));
  return bpl::object(bpl::handle<>(a));
}

// Calculate a domain corresponding to the given slice, assuming the given length.
// To be able to generate a domain we also need the parent block's lay
// The stride parameter is needed to account for non-unit-stride access (i.e. when
// slicing subblocks)
inline Domain<1> slice_to_domain(bpl::slice s, index_type offset, stride_type stride, length_type size)
{
  Py_ssize_t start, stop, step, length;
  int status = PySlice_GetIndicesEx((PySliceObject*)s.ptr(), size,
				    &start, &stop, &step, &length);
  return Domain<1>(offset + start*stride, step*stride, length);
}

template <typename T>
bpl::object subblock1(Block<1, T> &b, bpl::slice s)
{
  Domain<1> dom = slice_to_domain(s, b.offset(), b.stride(1, 0), b.size(1, 0));
  return bpl::object(std::shared_ptr<Block<1, T> >(new Block<1, T>(b, dom)));
}

template <typename T, typename M>
bpl::object subblock2(Block<2, T, M> &b, bpl::slice i, bpl::slice j)
{
  Domain<1> domi = slice_to_domain(i, b.offset(), b.stride(2, 0), b.size(2, 0));
  Domain<1> domj = slice_to_domain(j, 0, b.stride(2, 1), b.size(2, 1));
  return bpl::object(std::shared_ptr<Block<2, T, M> >(new Block<2, T, M>
    (b, Domain<2>(domi, domj))));
}

template <typename T, typename M>
bpl::object get_row(Block<2, T, M> &b, int i)
{
  if (i < 0) i += b.size(2, 0);
  Domain<1> domain(i * b.stride(2, 0), 1, b.size(2, 1));
  return bpl::object(std::shared_ptr<Block<1, T, M> >(new Block<1, T, M>(b, domain)));
}

template <typename T, typename M>
bpl::object get_col(Block<2, T, M> &b, int i)
{
  if (i < 0) i += b.size(2, 1);
  Domain<1> domain(i * b.stride(2, 1), b.stride(2, 0), b.size(2, 0));
  return bpl::object(std::shared_ptr<Block<1, T, M> >(new Block<1, T, M>(b, domain)));
}

template <typename T, typename M>
bpl::object diag(Block<2, T, M> &b)
{
  Domain<1> domain(0, b.stride(2, 0) + b.stride(2, 1), std::min(b.size(2, 0), b.size(2, 1)));
  return bpl::object(std::shared_ptr<Block<1, T, M> >(new Block<1, T, M>(b, domain)));
}

template <dimension_type D, typename T, typename M>
bpl::object real(Block<D, complex<T>, M> &b)
{
  return bpl::object(std::shared_ptr<Block<D, T, M> >
		     (new Block<D, T, M>(b, true)));
}

template <dimension_type D, typename T, typename M>
bpl::object imag(Block<D, complex<T>, M> &b)
{
  return bpl::object(std::shared_ptr<Block<D, T, M> >
		     (new Block<D, T, M>(b, false)));
}

template <dimension_type D, typename T, typename M>
void assign(Block<D, T, M> &b, Block<D, T, M> const &other)
{
  ovxx::assign<D>(b, other);
}

template <dimension_type D, typename T, typename M>
void assign_scalar(Block<D, T, M> &b, T val)
{
  ovxx::expr::Scalar<D, T> scalar(val);
  ovxx::assign<D>(b, scalar);
}

template <dimension_type D, typename T, typename M>
std::shared_ptr<Block<D, T, M> > copy(Block<D, T, M> const &b)
{
  Domain<D> dom = block_domain<D>(b);
  std::shared_ptr<Block<D, T, M> > other(new Block<D, T, M>(dom));
  ovxx::assign<D>(*other, b);
  return other;
}

template <dimension_type D, typename T, typename M>
bpl::object eq(Block<D, T, M> const &b1, Block<D, T, M> const &b2)
{
  typedef Block<D, T, M> B;
  typename ovxx::view_of<B>::const_type v1(const_cast<B&>(b1));
  typename ovxx::view_of<B>::const_type v2(const_cast<B&>(b2));
  Domain<D> dom = ovxx::block_domain<D>(b1);
  Block<D, bool, M> *result = new Block<D, bool, M>(dom);
  typename ovxx::view_of<Block<D, bool, M> >::type r(*result);
  r = vsip::eq(v1, v2);
  return bpl::object(std::shared_ptr<Block<D, bool, M> >(result));
}

template <dimension_type D, typename T, typename M>
bpl::object neg(Block<D, T, M> const &b)
{
  typedef Block<D, T, M> B;
  typename ovxx::view_of<B>::const_type v(const_cast<B&>(b));
  Domain<D> dom = ovxx::block_domain<D>(b);
  B *result = new B(dom);
  typename ovxx::view_of<B>::type r(*result);
  r = vsip::neg(v);
  return bpl::object(std::shared_ptr<B>(result));
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
void define_complex_subblocks(C &, T) {}

template <dimension_type D, typename C, typename T>
void define_complex_subblocks(C &block, complex<T>)
{
  block.def("real", real<D, T, typename C::wrapped_type::map_type>);
  block.def("imag", imag<D, T, typename C::wrapped_type::map_type>);
}

template <dimension_type D, typename T>
void define_block(char const *type_name)
{
  typedef Local_map M;
  typedef Block<D, T> block_type;
  typedef array_backref<block_type> backref_type;

  std::string backref_name(type_name);
  backref_name.append("_backref");
  bpl::class_<backref_type, std::shared_ptr<backref_type>, boost::noncopyable>
    backref(backref_name.c_str(), bpl::init<block_type&>());

  bpl::class_<block_type, std::shared_ptr<block_type>, boost::noncopyable> 
    block(type_name, bpl::no_init);
  block.setattr("dtype", get_dtype<T>());
  /// Conversion to array.
  block.def("__array__", make_array<D, T>);
  block.def("assign", assign<D, T, M>);
  block.def("assign", assign_scalar<D, T, M>);

  block.def("copy", copy<D, T, M>);

  block.def("size", total_size<D, T, M>);
  block.def("size", size<D, T, M>);
  block.add_property("shape", shape<D, T, M>);
  block.def("get", &traits<D, T>::get);
  block.def("put", &traits<D, T>::put);

  block.def("__eq__", eq<D, T, M>);
  block.def("__neg__", neg<D, T, M>);

  define_compound_assignment(block, T());
  define_complex_subblocks<D>(block, T());

  /// Construction from shape
  if (D != 1) // The case D==1 is covered in construct<T> below
    bpl::def("block", traits<D, T>::construct);
  bpl::def("block", traits<D, T>::construct_init);
  /// Conversion from array.
  bpl::def("block", construct<T>);
  /// Construct subblock
  bpl::def("subblock", subblock1<T>);
  if (D == 2)
  {
    bpl::def("subblock", subblock2<T, M>);
    block.def("row", get_row<T, M>);
    block.def("col", get_col<T, M>);
    block.def("diag", diag<T, M>);
  }

}

}


#endif

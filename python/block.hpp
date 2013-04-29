/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   VSIPL++ Library: Block adapter for NumPy's Array type.

#ifndef block_hpp_
#define block_hpp_

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <vsip/support.hpp>
#include <vsip/core/parallel/local_map.hpp>
#include <vsip/layout.hpp>
#include <vsip/dda.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/block_fill.hpp>
#include <vsip/dense.hpp> // for dense_complex_format

namespace pyvsip
{
namespace
{
  struct array_importer
  {
    array_importer()
    { 
      import_array();
    }
  } array_importer_;
}


namespace bpl = boost::python;

#define THROW(TYPE, REASON)              \
{                                        \
  PyErr_SetString(PyExc_##TYPE, REASON); \
  throw bpl::error_already_set();        \
}

template <typename T> inline NPY_TYPES get_typenum();
template <> inline NPY_TYPES get_typenum<bool>() { return NPY_BOOL;}
template <> inline NPY_TYPES get_typenum<npy_byte>() { return NPY_BYTE;}
template <> inline NPY_TYPES get_typenum<npy_ubyte>() { return NPY_UBYTE;}
template <> inline NPY_TYPES get_typenum<npy_short>() { return NPY_SHORT;}
template <> inline NPY_TYPES get_typenum<npy_ushort>() { return NPY_USHORT;}
template <> inline NPY_TYPES get_typenum<npy_int>() { return NPY_INT;}
template <> inline NPY_TYPES get_typenum<npy_uint>() { return NPY_UINT;}
template <> inline NPY_TYPES get_typenum<npy_long>() { return NPY_LONG;}
template <> inline NPY_TYPES get_typenum<npy_ulong>() { return NPY_ULONG;}
template <> inline NPY_TYPES get_typenum<npy_longlong>() { return NPY_LONGLONG;}
template <> inline NPY_TYPES get_typenum<npy_ulonglong>() { return NPY_ULONGLONG;}
template <> inline NPY_TYPES get_typenum<npy_float>() { return NPY_FLOAT;}
template <> inline NPY_TYPES get_typenum<npy_double>() { return NPY_DOUBLE;}
template <> inline NPY_TYPES get_typenum<npy_longdouble>() { return NPY_LONGDOUBLE;}
template <> inline NPY_TYPES get_typenum<npy_cfloat>() { return NPY_CFLOAT;}
template <> inline NPY_TYPES get_typenum<npy_cdouble>() { return NPY_CDOUBLE;}
template <> inline NPY_TYPES get_typenum<npy_clongdouble>() { return NPY_CLONGDOUBLE;}
template <> inline NPY_TYPES get_typenum<std::complex<float> >() { return NPY_CFLOAT;}
template <> inline NPY_TYPES get_typenum<std::complex<double> >() { return NPY_CDOUBLE;}
template <> inline NPY_TYPES get_typenum<std::complex<long double> >() { return NPY_CLONGDOUBLE;}
template <> inline NPY_TYPES get_typenum<bpl::object>() { return NPY_OBJECT;}
template <> inline NPY_TYPES get_typenum<bpl::handle<> >() { return NPY_OBJECT;}

// FIXME: We should probably define converters to map Index<1> to int,
// and Index<2> to tuple
template <> inline NPY_TYPES get_typenum<vsip::Index<1> >() { return NPY_UINT;}
template <> inline NPY_TYPES get_typenum<vsip::Index<2> >() { return NPY_OBJECT;}

// Due to a mismatch between the C and Python representations for integral types,
// we can't use a one-to-one map. Instead, we compare size and signeness.
template <class T>
inline
bool is_storage_compatible(PyObject *a)
{
  NPY_TYPES typenum = NPY_TYPES(PyArray_TYPE(a));

  if (boost::is_integral<T>::value && PyArray_ISINTEGER(a))
  {
    return (sizeof(T) == PyArray_ITEMSIZE(a)
	    && bool(boost::is_signed<T>::value) == bool(PyArray_ISSIGNED(a)));
  }
  else if (typenum == NPY_BOOL && (boost::is_same<T, signed char>::value ||
				   boost::is_same<T, unsigned char>::value))
  {
    return (sizeof(T) == PyArray_ITEMSIZE(a)
	    && bool(boost::is_signed<T>::value) == bool(PyArray_ISSIGNED(a)));
  }
  else
    return typenum == get_typenum<T>();
}

enum dimension_order { row, column};

template <vsip::dimension_type D, typename T>
class Block : public vsip::impl::Ref_count<Block<D, T> >
{
public:
  static vsip::dimension_type const dim = D;
  typedef T value_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;

  typedef vsip::Layout<dim, vsip::tuple<0,1,2>,
		       vsip::dense,
		       vsip::interleaved_complex> layout_type;

  typedef vsip::Local_map map_type;

  typedef value_type *ptr_type;
  typedef value_type const *const_ptr_type;

  /// Create a new standalone Block which owns its data.
  Block(vsip::Domain<dim> const &dom, map_type const &map,
	dimension_order o = row)
    : map_(map)
  {
    npy_intp dims[dim];
    for (vsip::dimension_type d = 0; d != dim; ++d)
      dims[d] = dom[d].length();
    int flags = o == row ? NPY_CARRAY : NPY_FARRAY;
    array_ = bpl::handle<>(PyArray_New(&PyArray_Type,
				       dim, dims, get_typenum<value_type>(),
				       0, 0, 0, flags, 0));
  }

  /// Create a new standalone Block which owns its data.
  Block(vsip::Domain<dim> const &dom, value_type value,
	map_type const &map,
	dimension_order o = row)
    : map_(map)
  {
    npy_intp dims[dim];
    for (vsip::dimension_type d = 0; d != dim; ++d)
      dims[d] = dom[d].length();
    int flags = o == row ? NPY_CARRAY : NPY_FARRAY;
    array_ = bpl::handle<>(PyArray_New(&PyArray_Type,
				       dim, dims, get_typenum<value_type>(),
				       0, 0, 0, flags, 0));
    vsip::impl::Block_fill<dim, Block>::exec(*this, value);
  }

  /// Create a new Block that is a subblock of another block.
  /// The start address of the subblock is offset by a fixed precomputed amount.
  /// dom 'first' members are ignored.
  template <vsip::dimension_type D1, typename T1>
  Block(Block<D1, T1> &base, vsip::length_type offset, vsip::Domain<dim> const &dom)
    : map_(base.map())
  {
    npy_intp strides[dim];
    npy_intp dims[dim];
    for (vsip::dimension_type d = 0; d != dim; ++d)
    {
      // numpy.ndarray counts strides in bytes, not elements.
      strides[d] = dom[d].stride() * itemsize();
      dims[d] = dom[d].length();
    }
    array_ = bpl::handle<>(PyArray_New(&PyArray_Type,
				       dim, dims, get_typenum<value_type>(),
				       strides, base.ptr() + offset,
				       0, NPY_BEHAVED, 0));
    /// Register the dependency.
    PyArray_BASE(array_.get()) = bpl::xincref(base.array().get());
  }

  /// Create a new block from a numpy.ndarray.
  Block(bpl::object o)
    : array_(bpl::xincref(o.ptr()))
  {
    if (!PyArray_Check(o.ptr()))
      THROW(TypeError, "argument is not a numpy array");
    if (!is_storage_compatible<T>(o.ptr()))
      THROW(TypeError, "argument is numpy array of wrong type");
    if (PyArray_NDIM(o.ptr()) != static_cast<npy_intp>(dim))
      THROW(ValueError, "argument array has incompatible shape");
    if (!PyArray_CHKFLAGS(o.ptr(), NPY_ALIGNED))
      THROW(ValueError, "argument array is not aligned");
    if (PyArray_CHKFLAGS(o.ptr(), NPY_NOTSWAPPED))
      THROW(ValueError, "argument array does not have native endianness");
    if (PyArray_ITEMSIZE(o.ptr()) != sizeof(T))
      THROW(ValueError, "itemsize does not match size of target type");
  }
  vsip::impl::Ref_counted_ptr<Block> copy() const
  {
    bpl::handle<> cp(PyArray_NewCopy(reinterpret_cast<PyArrayObject *>(array_.get()), NPY_ANYORDER));
    return vsip::impl::Ref_counted_ptr<Block>(new Block(bpl::object(cp)));
  }

  vsip::length_type size() const 
  { 
    vsip::length_type total = 1;
    for (vsip::dimension_type i = 0; i < dim; ++i) total *= size(dim, i);
    return total;
  }
  vsip::length_type size(vsip::dimension_type, vsip::dimension_type d) const 
  { return PyArray_DIM(array_.get(), d);}
  map_type const &map() const { return map_;}

  value_type get(vsip::index_type i) const
  { return *reinterpret_cast<value_type*>(PyArray_GETPTR1(array_.get(), i));}
  value_type get(vsip::index_type i, vsip::index_type j) const
  { return *reinterpret_cast<value_type*>(PyArray_GETPTR2(array_.get(), i, j));}
  value_type get(vsip::index_type i, vsip::index_type j, vsip::index_type k) const
  { return *reinterpret_cast<value_type*>(PyArray_GETPTR3(array_.get(), i, j, k));}

  void put(vsip::index_type i, value_type v)
  { *reinterpret_cast<value_type*>(PyArray_GETPTR1(array_.get(), i)) = v;}
  void put(vsip::index_type i, vsip::index_type j, value_type v)
  { *reinterpret_cast<value_type*>(PyArray_GETPTR2(array_.get(), i, j)) = v;}
  void put(vsip::index_type i, vsip::index_type j, vsip::index_type k, value_type v)
  { *reinterpret_cast<value_type*>(PyArray_GETPTR3(array_.get(), i, j, k)) = v;}

  ptr_type ptr()
  { return reinterpret_cast<ptr_type>(PyArray_DATA(array_.get()));}
  const_ptr_type ptr() const
  { return reinterpret_cast<const_ptr_type>(PyArray_DATA(array_.get()));}
  vsip::stride_type stride(vsip::dimension_type, vsip::dimension_type i) const
  { return stride(i) / itemsize();}

  bpl::handle<> array() const { return array_;}

private:
  // The following are useful wrappers around numpy.ndarray, but don't correspond to
  // Block methods, and thus are hidden from public view.
  npy_intp const *dims() const { return PyArray_DIMS(array_.get());}
  npy_intp const *strides() const { return PyArray_STRIDES(array_.get());}
  npy_intp stride(npy_intp i) const { return PyArray_STRIDE(array_.get(), i);}
  npy_intp itemsize() const { return sizeof(T);}
  bool writable() const { return PyArray_ISWRITEABLE(array_.get());}
  
  void reshape(int ndim, const npy_intp *dims, NPY_ORDER order=NPY_CORDER)
  {
    PyArray_Dims d = { const_cast<npy_intp *>(dims), ndim};
    array_ = bpl::handle<>(PyArray_Newshape((PyArrayObject *) array_.get(), &d, order));
  }

  bpl::handle<> array_;
  map_type map_;
};

} // namespace pyvsip

namespace vsip
{
namespace impl
{
/// pyvsip::Block provides shared pointer semantics for numpy.ndarrays, so should itself be
/// stored by-value.
// template <dimension_type D, typename T>
// struct View_block_storage<pyvsip::Block<D, T> > : By_value_block_storage<pyvsip::Block<D, T> >
// {};

} // namespace vsip::impl

template <dimension_type D, typename T>
struct get_block_layout<pyvsip::Block<D, T> >
{
  static dimension_type const dim = D;

  typedef typename pyvsip::Block<D, T>::layout_type type;
  typedef typename type::order_type order_type;
  static pack_type const packing = type::packing;
  static storage_format_type const storage_format = type::storage_format;
};

// template <dimension_type D, typename T>
// struct supports_dda<pyvsip::Block<D, T> >
// { static bool const value = true;};

} // namespace vsip


#endif

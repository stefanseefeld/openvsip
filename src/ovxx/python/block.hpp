//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_python_block_hpp_
#define ovxx_python_block_hpp_

#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#include <ovxx/python/config.hpp>
#include <numpy/arrayobject.h>
#include <vsip/dense.hpp>
#include <ovxx/assign.hpp>

namespace boost { namespace python {

// Help bpl::handle convert between PyArrayObject* and PyObject*
template <>
struct base_type_traits<PyArrayObject>
{
  typedef PyObject type;
};

}} // namespace boost::python

namespace ovxx
{
namespace python
{

namespace bpl = boost::python;
using boost::shared_ptr;

// Send all traces from OVXX_TRACE to the Python logger
inline void trace(char const *format, ...)
{
  bpl::object logging = bpl::import("logging");
  bpl::object info = logging.attr("info");
  va_list args;
  char msg[128];
  va_start(args, format);
  vsnprintf(msg, sizeof(msg), format, args);
  va_end(args);
  info(msg);
}

inline void initialize() { import_array();}

#define PYVSIP_THROW(TYPE, REASON)	 \
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
template <> inline NPY_TYPES get_typenum<complex<float> >() { return NPY_CFLOAT;}
template <> inline NPY_TYPES get_typenum<complex<double> >() { return NPY_CDOUBLE;}
template <> inline NPY_TYPES get_typenum<bpl::object>() { return NPY_OBJECT;}
template <> inline NPY_TYPES get_typenum<bpl::handle<> >() { return NPY_OBJECT;}

// FIXME: We should probably define converters to map Index<1> to int,
// and Index<2> to tuple
template <> inline NPY_TYPES get_typenum<Index<1> >() { return NPY_UINT;}
template <> inline NPY_TYPES get_typenum<Index<2> >() { return NPY_OBJECT;}

// Due to a mismatch between the C and Python representations for integral types,
// we can't use a one-to-one map. Instead, we compare size and signeness.
template <class T>
inline
bool is_storage_compatible(PyArrayObject const *a)
{
  NPY_TYPES typenum = NPY_TYPES(PyArray_TYPE(a));
  size_t itemsize = PyArray_ITEMSIZE(a);
  //  int typenum = PyArray_MinScalarType(const_cast<PyArrayObject*>(a))->type_num;

  if (is_integral<T>::value && PyTypeNum_ISINTEGER(typenum))
  {
    return (sizeof(T) == itemsize &&
	    bool(is_signed<T>::value) == PyTypeNum_ISSIGNED(typenum));
  }
  else if (PyTypeNum_ISBOOL(typenum) &&
	   (is_same<T, signed char>::value ||
	    is_same<T, unsigned char>::value))
  {
    return (sizeof(T) == itemsize &&
	    bool(is_signed<T>::value) == PyTypeNum_ISSIGNED(typenum));
  }
  else
    return typenum == get_typenum<T>();
}

enum dimension_order { row, column};

template <dimension_type D, typename T, typename M = Local_map> class Block;

template <dimension_type D, typename T, typename M>
class Block : public parallel::distributed_block<Block<D, T>, M>
{
  typedef parallel::distributed_block<Block<D, T>, M> base_type;
  typedef typename base_type::uT uT;

public:
  typedef M map_type;

  /// Create a new standalone Block which owns its data.
  Block(Domain<D> const &dom, M const &map)
    : base_type(dom, map)
  {}

  /// Create a new standalone Block which owns its data.
  Block(Domain<D> const &dom, T value, M const &map)
    : base_type(dom, map)
  {}
};

// This type implements the VSIPL++ block concept, and can
// thus be passed as a block parameter to any VSIPL++ operation.
// But its layout is not a compile-time constant, and the block is
// thus not particularly efficient.
// However, if the layout happens to match, the object may hold
// a much more efficient block representation internally, which can
// be checked for and accessed as "dense", in which case performance
// is much improved (including minimal-copy heterogeneous storage operations).
template <dimension_type D, typename T>
class Block<D, T>
{
  typedef Dense<D, T> block_type;
public:
  static dimension_type const dim = D;
  typedef T value_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;

  typedef Layout<dim, tuple<0,1,2>, vsip::dense, vsip::array> layout_type;

  typedef Local_map map_type;

  typedef value_type *ptr_type;
  typedef value_type const *const_ptr_type;

  /// Create a new empty (uninitialized) block.
  Block(Domain<dim> const &dom, dimension_order o = row)
    : block_(new block_type(dom)), array_(init_array(*block_))
  {}

  /// Create a new block with initial value.
  Block(Domain<dim> const &dom, value_type value, dimension_order o = row)
    : block_(new block_type(dom)), array_(init_array(*block_))
  {
    ovxx::assign<dim>(block_, expr::Scalar<dim, T>(value));
  }

  /// Create a block from an existing C++ block.
  /// (Used for to_python conversion.)
  Block(block_type &block) : block_(&block), array_(init_array(*block_))
  {}

  /// Create a new Subblock of a parent block and a (sub-)domain.
  /// Note that T1 != T for real and imag subblocks of complex parent blocks.
  template <dimension_type D1, typename T1>
  Block(Block<D1, T1> &base, length_type offset, Domain<dim> const &dom)
  {
    npy_intp dims[dim];
    npy_intp strides[dim];
    for (dimension_type d = 0; d != dim; ++d)
    {
      dims[d] = dom[d].length();
      strides[d] = dom[d].stride() * itemsize();
    }
    array_ = bpl::handle<PyArrayObject>
      ((PyArrayObject *)
       PyArray_New(&PyArray_Type, dim, dims, get_typenum<value_type>(),
		   strides, reinterpret_cast<value_type*>(base.ptr()) + offset,
		   0, NPY_ARRAY_BEHAVED, 0));
    /// Register the dependency.
    PyArray_SetBaseObject(array_.get(), base.array().release());
  }

  /// Create a new block wrapping a numpy.ndarray.
  /// This constructor is not exposed to Python, so we don't need to
  /// do any checks here. (See the pyvsip::block() function below.)
  Block(bpl::object o)
    : array_(reinterpret_cast<PyArrayObject *>(bpl::xincref(o.ptr()))) {}

  // Create a new block with the same shape and value-type as this,
  // but always using dense storage, no matter the storage of this block.
  shared_ptr<Block> copy() const
  {
    Domain<dim> dom = block_domain<dim>(*this);
    shared_ptr<Block> other(new Block(dom));
    ovxx::assign<dim>(*other, *this);
    return other;
  }

  // If this block is dense, operations may access it in terms of a stored_block,
  // which optimizes many operations (and avoids copies...)
  bool is_dense() const { return block_.get();}
  block_type &dense() { return *block_;}

  length_type size() const 
  { 
    length_type total = 1;
    for (dimension_type i = 0; i < dim; ++i) total *= size(dim, i);
    return total;
  }
  length_type size(dimension_type, dimension_type d) const 
  { return PyArray_DIM(array_.get(), d);}

  // Note: None of the 'get' and 'put' functions below validate their arguments.
  //       They may be called internally during assignment operations, which is assumed
  //       not to generate out-of-bound indices.
  //       For direct calls from Python code these functions are wrapped (see
  //       block_api.hpp) in error-checking functions.

  value_type get(index_type i) const
  {
    sync();
    return *reinterpret_cast<value_type*>(PyArray_GETPTR1(array_.get(), i));
  }
  value_type get(index_type i, index_type j) const
  {
    sync();
    return *reinterpret_cast<value_type*>(PyArray_GETPTR2(array_.get(), i, j));
  }
  value_type get(index_type i, index_type j, index_type k) const
  {
    sync();
    return *reinterpret_cast<value_type*>(PyArray_GETPTR3(array_.get(), i, j, k));
  }

  void put(index_type i, value_type v)
  {
    sync();
    *reinterpret_cast<value_type*>(PyArray_GETPTR1(array_.get(), i)) = v;
  }
  void put(index_type i, index_type j, value_type v)
  {
    sync();
    *reinterpret_cast<value_type*>(PyArray_GETPTR2(array_.get(), i, j)) = v;
  }
  void put(index_type i, index_type j, index_type k, value_type v)
  {
    sync();
    *reinterpret_cast<value_type*>(PyArray_GETPTR3(array_.get(), i, j, k)) = v;
  }

  Block &operator +=(Block const &o)
  {
    typename view_of<Block>::type self(*this);
    typename view_of<Block>::const_type other(const_cast<Block&>(o));
    self += other;
    return *this;
  }
  Block &operator +=(value_type s)
  {
    typename view_of<Block>::type self(*this);
    typedef expr::Scalar<dim, value_type> S;
    S scalar(s);
    typename view_of<S>::const_type other(scalar);
    self += other;
    return *this;
  }
  Block &operator -=(Block const &o)
  {
    typename view_of<Block>::type self(*this);
    typename view_of<Block>::const_type other(const_cast<Block&>(o));
    self -= other;
    return *this;
  }
  Block &operator -=(value_type s)
  {
    typename view_of<Block>::type self(*this);
    typedef expr::Scalar<dim, value_type> S;
    S scalar(s);
    typename view_of<S>::const_type other(scalar);
    self -= other;
    return *this;
  }
  Block &operator *=(Block const &o)
  {
    typename view_of<Block>::type self(*this);
    typename view_of<Block>::const_type other(const_cast<Block&>(o));
    self *= other;
    return *this;
  }
  Block &operator *=(value_type s)
  {
    typename view_of<Block>::type self(*this);
    typedef expr::Scalar<dim, value_type> S;
    S scalar(s);
    typename view_of<S>::const_type other(scalar);
    self *= other;
    return *this;
  }
  Block &operator /=(Block const &o)
  {
    typename view_of<Block>::type self(*this);
    typename view_of<Block>::const_type other(const_cast<Block&>(o));
    self /= other;
    return *this;
  }
  Block &operator /=(value_type s)
  {
    typename view_of<Block>::type self(*this);
    typedef expr::Scalar<dim, value_type> S;
    S scalar(s);
    typename view_of<S>::const_type other(scalar);
    self /= other;
    return *this;
  }

  ptr_type ptr()
  { return reinterpret_cast<ptr_type>(PyArray_DATA(array_.get()));}
  const_ptr_type ptr() const
  { return reinterpret_cast<const_ptr_type>(PyArray_DATA(array_.get()));}
  vsip::stride_type stride(dimension_type, dimension_type i) const
  { return stride(i) / itemsize();}

  bpl::handle<> array() const { sync(); return array_;}

private:
  static bpl::handle<PyArrayObject> init_array(block_type &block)
  {
    npy_intp dims[dim];
    for (dimension_type d = 0; d != dim; ++d)
      dims[d] = block.size(dim, d);
    // int flags = o == row ? NPY_CORDER : NPY_FORTRANORDER;
    int flags = NPY_CORDER;
    return bpl::handle<PyArrayObject>
      ((PyArrayObject *)
       PyArray_New(&PyArray_Type, dim, dims, get_typenum<value_type>(),
		   0, block.ptr(), 0, flags, 0));
  }
  // make sure data in host memory is valid
  void sync() const { if (block_.get()) block_->ptr();}

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
    array_ = bpl::handle<PyArrayObject>(PyArray_Newshape(array_.get(), &d, order));
  }

  // block_ is only used if this object owns its data, i.e. is not
  // merely a subblock of a parent block.
  typename block_traits<block_type>::ptr_type block_;
  bpl::handle<PyArrayObject> array_;
};

} // namespace ovxx::python

template <dimension_type D, typename T>
struct distributed_local_block<python::Block<D, T> >
{
  typedef python::Block<D, T> type;
  typedef python::Block<D, T> proxy_type;
};

template <dimension_type D, typename T, typename M>
struct distributed_local_block<python::Block<D, T, M> >
{
  typedef python::Block<D, T> type;
  typedef typename python::Block<D, T, M>::proxy_local_block_type proxy_type;
};

template <dimension_type D, typename T>
python::Block<D, T> &get_local_block(python::Block<D, T> &block) { return block;}

template <dimension_type D, typename T>
python::Block<D, T> const &get_local_block(python::Block<D, T> const &block) { return block;}

template <dimension_type D, typename T, typename M>
inline typename python::Block<D, T> &
get_local_block(python::Block<D, T, M> &block) { return block.get_local_block();}

template <dimension_type D, typename T, typename M>
inline typename python::Block<D, T> &
get_local_block(python::Block<D, T, M> const &block) { return block.get_local_block();}

template <dimension_type D, typename T>
struct block_traits<python::Block<D, T> > : by_value_traits<python::Block<D, T> >
{};

} // namespace ovxx

namespace vsip
{
template <dimension_type D, typename T>
struct get_block_layout<ovxx::python::Block<D, T> >
{
  static dimension_type const dim = D;

  typedef typename ovxx::python::Block<D, T>::layout_type type;
  typedef typename type::order_type order_type;
  static pack_type const packing = type::packing;
  static storage_format_type const storage_format = type::storage_format;
};

template <dimension_type D, typename T>
struct supports_dda<ovxx::python::Block<D, T> >
{ static bool const value = true;};

} // namespace vsip

#endif

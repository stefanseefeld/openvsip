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
#include <ovxx/support.hpp>
#include <vsip/domain.hpp>
#include <ovxx/layout.hpp>
#include <ovxx/storage.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/view.hpp>
#include <vsip/impl/local_map.hpp>
#if OVXX_HAVE_OPENCL
#include <ovxx/opencl/dda.hpp>
#endif
#include <stdexcept>

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

// For float and double we need a storage wrapper,
// as we can't be sure what value-type our storage
// has, to support real-valued views into complex storage.
template <typename T>
class sm_proxy
{
  typedef shared_ptr<storage_manager<T> > smanager_ptr;
  typedef shared_ptr<storage_manager<complex<T> > > csmanager_ptr;
public:
  typedef T value_type;
  typedef T *ptr_type;
  typedef T const *const_ptr_type;
  typedef T &reference_type;
  typedef T const &const_reference_type;

  // the 'normal' constructor
  sm_proxy(allocator *a, length_type s, bool f)
    : parent_(new storage_manager<T>(a, s, f))
  {}
  // the subblock constructor
  sm_proxy(smanager_ptr sm) : parent_(sm) {}
  // the real/imag component constructor
  sm_proxy(csmanager_ptr sm, bool r) : cparent_(sm), real_(r) {}

  value_type get(index_type i) const
  {
    if (parent_) return parent_->get(i);
    else if (real_) return cparent_->at(i).real();
    else return cparent_->at(i).imag();
  }
  void put(index_type i, value_type value)
  {
    if (parent_) parent_->put(i, value);
    else if (real_) cparent_->at(i) = complex<T>(value, cparent_->at(i).imag());
    else cparent_->at(i) = complex<T>(cparent_->at(i).real(), value);
  }
  reference_type at(index_type i)
  {
    if (parent_) return parent_->at(i);
    else if (real_) return cparent_->at(i).real();
    else return cparent_->at(i).imag();
  }
  
  ptr_type ptr()
  {
    if (parent_) return parent_->ptr();
    else if (real_) return reinterpret_cast<T*>(cparent_->ptr());
    else return reinterpret_cast<T*>(cparent_->ptr()) + 1;
  }
#if OVXX_HAVE_OPENCL
  opencl::buffer buffer()
  {
    if (parent_) return parent_->buffer();
    else return cparent_->buffer();
  }
#endif
private:
  smanager_ptr parent_;
  csmanager_ptr cparent_;
  bool real_;
};

template <typename T,
	  bool need_proxy = is_same<T, float>::value || is_same<T, double>::value>
struct storage_traits;

template <typename T>
struct storage_traits<T, false>
{
  typedef storage_manager<T, ovxx::array> type;
};

template <typename T>
struct storage_traits<T, true>
{
  typedef sm_proxy<T> type;
};

template <dimension_type D, typename T, typename M = Local_map> class Block;

// This block is modeled after stored_block, but without support for user-storage.
// In addition, this block-type uses a runtime layout, as it is also used for
// subblocks, to avoid having to export many different block types to Python.
template <dimension_type D, typename T>
class Block<D, T, Local_map>
{
  typedef storage<T, ovxx::array> storage_type;
  typedef typename storage_traits<T>::type smanager_type;
  //  typedef storage_manager<T, ovxx::array> smanager_type;
  template <dimension_type D1, typename T1, typename M1>
  friend class Block;
public:
  typedef T value_type;
  typedef T &reference_type;
  typedef T const &const_reference_type;

  typedef Rt_layout<D> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;

  typedef typename smanager_type::ptr_type ptr_type;
  typedef typename smanager_type::const_ptr_type const_ptr_type;

  typedef Local_map map_type;

  static dimension_type const dim = D;

  /// Create a new empty (uninitialized) block.
  Block(Domain<dim> const &dom, map_type const &map = map_type())
    : offset_(0),
      layout_(Rt_layout<dim>(dense, tuple<>(), array), dom),
      smanager_(new smanager_type(map.impl_allocator(), layout_.total_size(), false)),
      map_(map),
      array_refcount_(0)
  {
    map_.impl_apply(dom);
  }

  /// Create a new block with initial value.
  Block(Domain<dim> const &dom, T value, map_type const &map = map_type())
    : offset_(0),
      layout_(Rt_layout<dim>(dense, tuple<>(), array), dom),
      smanager_(new smanager_type(map.impl_allocator(), layout_.total_size(), true)),
      map_(map),
      array_refcount_(0)
  {
    map_.impl_apply(dom);
    for (index_type i = 0; i < layout_.total_size(); ++i)
      smanager_->put(i, value);
  }

  // constructor used internally to form a subblock of parent.
  // dom holds physical coordinates, i.e. strides express distance in elements,
  // not rows / columns.
  template <dimension_type D1>
  Block(Block<D1, T> &parent, Domain<dim> const &dom)
    : offset_(0),
      layout_(dom),
      smanager_(parent.smanager_),
      array_refcount_(0)
  {
    for (index_type i = 0; i != dim; ++i) offset_ += dom[i].first();
  }
  // construct a real or imaginary component block from a complex
  // parent block
  Block(Block<D, complex<T> > &parent, bool real)
    : offset_(parent.offset_),
      layout_(parent.layout_),
      smanager_(new smanager_type(parent.smanager_, real)),
      array_refcount_(0)
  {}

  ~Block() {}

  void increment_count() const {}
  void decrement_count() const {}

  map_type const &map() const VSIP_NOTHROW { return map_;}

  length_type size() const VSIP_NOTHROW
  {
    length_type retval = layout_.size(0);
    for (dimension_type d = 1; d < dim; ++d)
      retval *= layout_.size(d);
    return retval;
  }

  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION((block_dim == 1 || block_dim == dim) && (d < block_dim));

    if (block_dim == 1) return size();
    else return layout_.size(d);
  }

  value_type get(index_type idx) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(idx < size());
    return smanager_->get(offset_ + layout_.index(idx));
  }
  value_type get(index_type idx0, index_type idx1) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(idx0 < layout_.size(0) && idx1 < layout_.size(1));
    return smanager_->get(offset_ + layout_.index(Index<2>(idx0, idx1)));
  }
  value_type get(index_type idx0, index_type idx1, index_type idx2) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(idx0 < layout_.size(0) && idx1 < layout_.size(1) &&
		      idx2 < layout_.size(2));
    return smanager_->get(offset_ + layout_.index(Index<3>(idx0, idx1, idx2)));
  }

  void put(index_type i, T v) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < size());
    smanager_->put(offset_ + layout_.index(i), v);
  }
  void put(index_type i, index_type j, T v) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1));
    smanager_->put(offset_ + layout_.index(Index<2>(i, j)), v);
  }
  void put(index_type i, index_type j, index_type k, T v)
    VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1) &&
		      k < layout_.size(2));
    smanager_->put(offset_ + layout_.index(Index<3>(i, j, k)), v);
  }

  reference_type ref(index_type i) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < size());
    return smanager_->at(offset_ + i);
  }
  const_reference_type ref(index_type i) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < size());
    return smanager_->at(offset_ + i);
  }
  reference_type ref(index_type i, index_type j)
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1));
    return smanager_->at(offset_ + layout_.index(Index<2>(i, j)));
  }
  const_reference_type ref(index_type i, index_type j) const
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1));
    return smanager_->at(offset_ + layout_.index(Index<2>(i, j)));
  }
  reference_type ref(index_type i, index_type j, index_type k)
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1) && k < layout_.size(2));
    return smanager_->at(offset_ + layout_.index(Index<3>(i, j, k)));
  }
  const_reference_type ref(index_type i, index_type j, index_type k) const
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1) && k < layout_.size(2));
    return smanager_->at(offset_ + layout_.index(Index<3>(i, j, k)));
  }

  index_type offset() const { return offset_;}
  stride_type stride(dimension_type block_dim, dimension_type d) const
  {
    OVXX_PRECONDITION(block_dim == 1 || block_dim == dim);
    OVXX_PRECONDITION(d < dim);
    return layout_.stride(d);
  }

#define OVXX_DEFINE_OP(OP)						\
  Block &operator OP(Block const &o)					\
  {									\
    typename view_of<Block>::type self(*this);				\
    typename view_of<Block>::const_type other(const_cast<Block&>(o));	\
    self OP other;							\
    return *this;							\
  }									\
  Block &operator OP(value_type s)					\
  {									\
    typename view_of<Block>::type self(*this);				\
    typedef expr::Scalar<dim, value_type> S;				\
    S scalar(s);							\
    typename view_of<S>::const_type other(scalar);			\
    self OP other;							\
    return *this;							\
  }
  OVXX_DEFINE_OP(+=)
  OVXX_DEFINE_OP(-=)
  OVXX_DEFINE_OP(*=)
  OVXX_DEFINE_OP(/=)

#undef OVXX_DEFINE_OP

  ptr_type ptr() { return smanager_->ptr() + offset_;}
  const_ptr_type ptr() const { return smanager_->ptr() + offset_;}
#if OVXX_HAVE_OPENCL
  opencl::buffer buffer() { return smanager_->buffer();}
  opencl::buffer buffer() const { return smanager_->buffer();}
#endif
  void ref_ptr() { ++array_refcount_;}
  void unref_ptr()
  {
    --array_refcount_;
    // TODO: Add check to actually pin the storage
    // if (!array_refcount_)
    //   std::cout << "block unlocked " << std::endl;
  }

private:
  length_type offset_;
  applied_layout_type layout_;
  shared_ptr<smanager_type> smanager_;
  map_type map_;
  unsigned array_refcount_;
};

template <dimension_type D, typename T, typename M>
class Block : public ovxx::parallel::distributed_block<Block<D, T>, M>
{
  typedef ovxx::parallel::distributed_block<Block<D, T>, M> base_type;
  typedef typename base_type::uT uT;
public:
  typedef M map_type;

  Block(Domain<D> const &dom, map_type const &map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, map)
    {}

  Block(Domain<D> const &dom, T value, map_type const &map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, value, map)
    {}
};

template <dimension_type D, typename T, typename M>
inline typename Block<D, T, M>::local_block_type &
get_local_block(Block<D, T, M> const &block)
{
  return block.get_local_block();
}


} // namespace ovxx::python

// This is needed to prevent the default "has_put" check
// which would attempt to instantiate an invalid 'put' overload
// and cause a compile error.
template <dimension_type D, typename T>
struct is_modifiable_block<python::Block<D, T> >
{
  static bool const value = true;
};

template <dimension_type D, typename T>
struct distributed_local_block<python::Block<D, T> >
{
  typedef python::Block<D, T> type;
  typedef python::Block<D, T> proxy_type;
};

template <dimension_type D, typename T, typename M>
struct distributed_local_block<python::Block<D, T, M> >
{
  typedef python::Block<D, T, Local_map> type;
  typedef typename python::Block<D, T, M>::proxy_local_block_type proxy_type;
};

#if OVXX_HAVE_OPENCL
namespace opencl
{
namespace detail
{
template <dimension_type D, typename T>
struct has_buffer<python::Block<D, T> >
{
  static bool const value = true;
};
} // namespace ovxx::opencl::detail
} // namespace ovxx::opencl
#endif
} // namespace ovxx

namespace vsip
{
template <dimension_type D, typename T>
struct get_block_layout<ovxx::python::Block<D, T> >
{
  static dimension_type const dim = D;

  typedef tuple<> order_type;
  static pack_type const packing = dense;
  static storage_format_type const storage_format = array;
  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <dimension_type D, typename T>
struct supports_dda<ovxx::python::Block<D, T> >
{ static bool const value = true;};

} // namespace vsip


#endif

/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/

/// Description
///   C-VSIPL API: block API.

#ifndef vsip_csl_cvsip_block_hpp_
#define vsip_csl_cvsip_block_hpp_

#include <vsip/support.hpp>
#include <vsip/core/parallel/local_map.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/dda.hpp>
#include <vsip/core/block_traits.hpp>

namespace vsip_csl
{
namespace cvsip
{
namespace impl
{
template <typename T>
T *offset(T *d, vsip_offset o) { return d + o;}

template <typename T>
T const *offset(T const *d, vsip_offset o) { return d + o;}

template <typename T>
std::pair<T *, T *> offset(std::pair<T *, T *> d, vsip_offset o)
{ return std::pair<T *, T *>(d.first + o, d.second + o);}

template <typename T>
std::pair<T const *, T const *> offset(std::pair<T const *, T const *> d, vsip_offset o)
{ return std::pair<T const *, T const *>(d.first + o, d.second + o);}

} // namespace cvsip::impl


// This is a block according to the requirements in
// 6.1 of the VSIPL++ spec, and thus is suitable to be passed
// to VSIPL++ operations.
//
// This block acts as a proxy to the storage that is owned by the C-VSIPL
// vsip_block. The latter uses Dense<>, which cares for synching
// (admitting, releasing) user-storage. Here we access Dense<>::ptr(),
// which is assumed to be stable over the storage's lifetime.
//
// Special care has to be taken for 'derived' blocks (real blocks referring
// to the real and imaginary parts of a complex block):
// As their layout parameters (offset, stride, length) refer to the
// (appropriately cast) parent block, we need to keep another set of those 
// parameters that is user-visible. Each time a user modifies those we
// recompute the 'real' ones.
// However, as this conversion only has to happen if the storage type of the
// parent block is interleaved.
template <vsip::dimension_type D, typename S,
          bool may_be_derived = !vsip::impl::is_complex<typename S::value_type>::value &&
#if !VSIP_IMPL_PREFER_SPLIT_COMPLEX
                                true
#else
// Things are much easier to handle with split complex storage.
                                false
#endif
         >
class Block;

template <typename S>
class Block<1, S, false>
{
public:
  static vsip::dimension_type const dim = 1;
  typedef typename S::value_type value_type;
  typedef value_type &           reference_type;
  typedef value_type const &     const_reference_type;
  typedef vsip::Local_map        map_type;

  typedef typename S::ptr_type ptr_type;
  typedef typename S::const_ptr_type const_ptr_type;

  Block(vsip_offset o, vsip_stride s, vsip_length l, S &storage,
        bool /*derived*/)
    : storage_(storage), offset_(o), stride_(s), length_(l) {}
  ~Block() {}

  void increment_count() const {}
  void decrement_count() const {}

  bool is_derived() const { return false;}
  vsip_offset offset() const { return offset_;}
  void offset(vsip_offset o) { offset_ = o;}
  vsip_stride stride() const { return stride_;}
  void stride(vsip_stride s) { stride_ = s;}
  vsip_length length() const { return length_;}
  void length(vsip_length l) { length_ = l;}

  vsip::length_type size() const { return length_;}
  vsip::length_type size(vsip::dimension_type, vsip::dimension_type) const
  { return size();}
  map_type const &map() const { return map_;}

  value_type get(vsip::index_type i) const
  { return storage_.get(offset_ + stride_ * i);}
  void put(vsip::index_type i, value_type v)
  { storage_.put(offset_ + stride_ * i, v);}

  // support direct data access protocol
  ptr_type ptr()  { return impl::offset(storage_.ptr(), offset_);}
  const_ptr_type ptr() const { return impl::offset(storage_.ptr(), offset_);}
  vsip::stride_type stride(vsip::dimension_type, vsip::dimension_type) const 
  { return stride_;}

private:
  S &                 storage_;
  vsip_offset         offset_;
  vsip_stride         stride_;
  vsip_length         length_;
  map_type            map_;
};

template <typename S>
class Block<1, S, true>
{
  struct Derived_parameters 
  {
    vsip_offset offset;
    vsip_stride stride;
  };
public:
  static vsip::dimension_type const dim = 1;
  typedef typename S::value_type value_type;
  typedef value_type &           reference_type;
  typedef value_type const &     const_reference_type;
  typedef vsip::Local_map        map_type;

  typedef typename S::ptr_type ptr_type;
  typedef typename S::const_ptr_type const_ptr_type;

  Block(vsip_offset o, vsip_stride s, vsip_length l, S &storage,
        bool derived)
    : storage_(storage), offset_(o), stride_(s), length_(l),
      derived_params_(0)
  {
    if (derived)
    {
      derived_params_ = new Derived_parameters;
      derived_params_->offset = offset_;
      derived_params_->stride = stride_;
      // For interleaved adjust units.
      offset_ *= 2;
      stride_ *= 2;
    }
  }
  ~Block() { delete derived_params_;}

  void increment_count() const {}
  void decrement_count() const {}

  bool is_derived() const { return derived_params_;}

  // As these are public accessors we need to convert appropriately.
  vsip_offset offset() const
  { return derived_params_ ? derived_params_->offset : offset_;}
  void offset(vsip_offset o)
  {
    offset_ = o;
    if (derived_params_)
    {
      derived_params_->offset = offset_;
      offset_ *= 2;
    }
  }
  vsip_stride stride() const
  { return derived_params_ ? derived_params_->stride : stride_;}
  void stride(vsip_stride s)
  {
    stride_ = s;
    if (derived_params_)
    {
      derived_params_->stride = stride_;
      stride_ *= 2;
    }
  }
  vsip_length length() const { return length_;}
  void length(vsip_length l) { length_ = l;}

  // Report the actual size, not the public one.
  vsip::length_type size() const { return length_;}
  vsip::length_type size(vsip::dimension_type, vsip::dimension_type) const
  { return size();}
  map_type const &map() const { return map_;}

  value_type get(vsip::index_type i) const
  { return storage_.get(offset_ + stride_ * i);}
  void put(vsip::index_type i, value_type v)
  { storage_.put(offset_ + stride_ * i, v);}

  // support direct data access protocol
  ptr_type ptr() { return impl::offset(storage_.ptr(), offset_);}
  const_ptr_type ptr() const { return impl::offset(storage_.ptr(), offset_);}
  vsip::stride_type stride(vsip::dimension_type, vsip::dimension_type) const 
  { return stride_;}

private:
  S &                 storage_;
  vsip_offset         offset_;
  vsip_stride         stride_;
  vsip_length         length_;
  Derived_parameters *derived_params_; // only waste a pointer
  map_type            map_;
};

template <typename S>
class Block<2, S, false>
{
public:
  static vsip::dimension_type const dim = 2;
  typedef typename S::value_type value_type;
  typedef value_type &           reference_type;
  typedef value_type const &     const_reference_type;
  typedef vsip::Local_map        map_type;

  typedef typename S::ptr_type ptr_type;
  typedef typename S::const_ptr_type const_ptr_type;

  Block(vsip_offset o,
        vsip_stride c, vsip_length m,
        vsip_stride r, vsip_length n,
        S &storage,
        bool /*derived*/)
    : storage_(storage), offset_(o),
      col_stride_(c), row_stride_(r),
      rows_(m), cols_(n)
  {}
  ~Block() {}

  void increment_count() const {}
  void decrement_count() const {}

  bool is_derived() const { return false;}
  // For split complex there is no difference between the public
  // and the internal layout.
  vsip_offset offset() const { return offset_;}
  void offset(vsip_offset o) { offset_ = o;}
  vsip_stride col_stride() const { return col_stride_;}
  void col_stride(vsip_stride s) { col_stride_ = s;}
  vsip_stride row_stride() const { return row_stride_;}
  void row_stride(vsip_stride s) { row_stride_ = s;}
  vsip_length rows() const { return rows_;}
  void rows(vsip_length r) { rows_ = r;}
  vsip_length cols() const { return cols_;}
  void cols(vsip_length c) { cols_ = c;}

  vsip::length_type size() const { return rows_*cols_;}
  vsip::length_type size(vsip::dimension_type D, vsip::dimension_type d) const
  { return D==1 ? size() : d == 0 ? rows() : cols();}
  map_type const &map() const { return map_;}

  value_type get(vsip::index_type r, vsip::index_type c) const 
  { return storage_.get(offset_ + col_stride_ * r + row_stride_ * c);}
  value_type get(vsip::index_type i) const { return get(i/cols(), i % cols());}
  void put(vsip::index_type r, vsip::index_type c, value_type v)
  {
    assert(r < rows() && c < cols());
    assert(storage_.size() > offset_ + col_stride_ * r + row_stride_ * c);
    storage_.put(offset_ + col_stride_ * r + row_stride_ * c, v);
  }
  void put(vsip::index_type i, value_type v) { put(i/cols(), i % cols(), v);}

  // support direct data access protocol
  ptr_type ptr() { return impl::offset(storage_.ptr(), offset_);}
  const_ptr_type ptr() const { return impl::offset(storage_.ptr(), offset_);}
  vsip::stride_type stride(vsip::dimension_type D, vsip::dimension_type d) const
  { return D == 1 ? row_stride_ : d == 0 ? col_stride_ : row_stride_;}

private:
  S &                 storage_;
  vsip_offset         offset_;
  // stride between elements in a column (and thus, between rows.)
  vsip_stride         col_stride_;
  // stride between elements in a row (and thus, between columns.
  vsip_stride         row_stride_;
  vsip_length         rows_;
  vsip_length         cols_;
  map_type            map_;
};

template <typename S>
class Block<2, S, true>
{
  struct Derived_parameters 
  {
    vsip_offset offset;
    vsip_stride col_stride;
    vsip_stride row_stride;
  };
public:
  static vsip::dimension_type const dim = 2;
  typedef typename S::value_type value_type;
  typedef value_type &           reference_type;
  typedef value_type const &     const_reference_type;
  typedef vsip::Local_map        map_type;

  typedef typename S::ptr_type ptr_type;
  typedef typename S::const_ptr_type const_ptr_type;

  Block(vsip_offset o,
        vsip_stride c, vsip_length m,
        vsip_stride r, vsip_length n,
        S &storage,
        bool derived)
    : storage_(storage), offset_(o),
      col_stride_(c), row_stride_(r),
      rows_(m), cols_(n),
      derived_params_(0)
  {
    if (derived)
    {
      derived_params_ = new Derived_parameters;
      derived_params_->offset = offset_;
      derived_params_->col_stride = col_stride_;
      derived_params_->row_stride = row_stride_;
      // For interleaved adjust units.
      offset_ *= 2;
      col_stride_ *= 2;
      row_stride_ *= 2;
    }
  }
  ~Block() { delete derived_params_;}

  void increment_count() const {}
  void decrement_count() const {}

  bool is_derived() const { return derived_params_;}

  // As these are public accessors we need to convert appropriately.
  vsip_offset offset() const
  { return derived_params_ ? derived_params_->offset : offset_;}
  void offset(vsip_offset o) 
  {
    offset_ = o;
    if (derived_params_)
    {
      derived_params_->offset = offset_;
      offset_ *= 2;
    }
  }
  vsip_stride col_stride() const
  { return derived_params_ ? derived_params_->col_stride : col_stride_;}
  void col_stride(vsip_stride s) 
  {
    col_stride_ = s;
    if (derived_params_)
    {
      derived_params_->col_stride = col_stride_;
      col_stride_ *= 2;
    }
  }
  vsip_stride row_stride() const
  { return derived_params_ ? derived_params_->row_stride : row_stride_;}
  void row_stride(vsip_stride s) 
  {
    row_stride_ = s;
    if (derived_params_)
    {
      derived_params_->row_stride = row_stride_;
      row_stride_ *= 2;
    }
  }
  vsip_length rows() const { return rows_;}
  void rows(vsip_length r) { rows_ = r;}
  vsip_length cols() const { return cols_;}
  void cols(vsip_length c) { cols_ = c;}

  vsip::length_type size() const { return rows_*cols_;}
  vsip::length_type size(vsip::dimension_type D, vsip::dimension_type d) const
  { return D==1 ? size() : d == 0 ? rows() : cols();}
  map_type const &map() const { return map_;}

  value_type get(vsip::index_type r, vsip::index_type c) const 
  { return storage_.get(offset_ + col_stride_ * r + row_stride_ * c);}
  value_type get(vsip::index_type i) const { return get(i/cols(), i % cols());}
  void put(vsip::index_type r, vsip::index_type c, value_type v)
  { storage_.put(offset_ + col_stride_ * r + row_stride_ * c, v);}
  void put(vsip::index_type i, value_type v) { put(i/cols(), i % cols(), v);}

  // support direct data access protocol
  ptr_type ptr() { return impl::offset(storage_.ptr(), offset_);}
  const_ptr_type ptr() const { return impl::offset(storage_.ptr(), offset_);}
  vsip::stride_type stride(vsip::dimension_type D, vsip::dimension_type d) const
  { return D == 1 ? row_stride_ : d == 0 ? col_stride_ : row_stride_;}

private:
  S &                 storage_;
  vsip_offset         offset_;
  // stride between elements in a column (and thus, between rows.)
  vsip_stride         col_stride_;
  // stride between elements in a row (and thus, between columns.
  vsip_stride         row_stride_;
  vsip_length         rows_;
  vsip_length         cols_;
  Derived_parameters *derived_params_;
  map_type            map_;
};

template <typename S>
class Block<3, S, false>
{
public:
  static vsip::dimension_type const dim = 3;
  typedef typename S::value_type value_type;
  typedef value_type &           reference_type;
  typedef value_type const &     const_reference_type;
  typedef vsip::Local_map        map_type;

  typedef typename S::ptr_type ptr_type;
  typedef typename S::const_ptr_type const_ptr_type;

  Block(vsip_offset o,
        vsip_stride zs, vsip_length zl,
        vsip_stride ys, vsip_length yl,
        vsip_stride xs, vsip_length xl,
        S &storage,
        bool /*derived*/)
    : storage_(storage), offset_(o),
      z_stride_(zs), y_stride_(ys), x_stride_(xs),
      z_length_(zl), y_length_(yl), x_length_(xl)
  {}
  ~Block() {}

  void increment_count() const {}
  void decrement_count() const {}

  bool is_derived() const { return false;}
  vsip_offset offset() const { return offset_;}
  void offset(vsip_offset o) { offset_ = o;}
  vsip_stride z_stride() const { return z_stride_;}
  void z_stride(vsip_stride s) { z_stride_ = s;}
  vsip_stride y_stride() const { return y_stride_;}
  void y_stride(vsip_stride s) { y_stride_ = s;}
  vsip_stride x_stride() const { return x_stride_;}
  void x_stride(vsip_stride s) { x_stride_ = s;}
  vsip_length z_length() const { return z_length_;}
  void z_length(vsip_length l) { z_length_ = l;}
  vsip_length y_length() const { return y_length_;}
  void y_length(vsip_length l) { y_length_ = l;}
  vsip_length x_length() const { return x_length_;}
  void x_length(vsip_length l) { x_length_ = l;}

  vsip::length_type size() const { return z_length_*y_length_*x_length_;}
  vsip::length_type size(vsip::dimension_type D, vsip::dimension_type d) const
  {
    if (D == 1) return size();
    else return d == 0 ? z_length_ : d == 1 ? y_length_ : x_length_;
  }
  map_type const &map() const { return map_;}

  value_type get(vsip::index_type h, vsip::index_type i, vsip::index_type j) const 
  { return storage_.get(offset_ + h * z_stride_ + i * y_stride_ + j * x_stride_);}
  void put(vsip::index_type h, vsip::index_type i, vsip::index_type j, value_type v)
  { storage_.put(offset_ + h * z_stride_ + i * y_stride_ + j * x_stride_, v);}
  void put(vsip::index_type i, value_type v) 
  {
    vsip::index_type planes = y_length() * x_length();
    vsip::index_type z = i / planes;
    vsip::index_type y = (i - z * planes) % x_length();
    vsip::index_type x = i - z * planes - y * x_length();
    put(z, y, x, v);
  }

  // support direct data access protocol
  ptr_type ptr() { return impl::offset(storage_.ptr(), offset_);}
  const_ptr_type ptr() const { return impl::offset(storage_.ptr(), offset_);}
  vsip::stride_type stride(vsip::dimension_type D, vsip::dimension_type d) const
  { return D == 1 ? z_stride_ : d == 0 ? z_stride_ : d == 1 ? y_stride_ : x_stride_;}

private:
  S &                 storage_;
  vsip_offset         offset_;
  vsip_stride         z_stride_;
  vsip_stride         y_stride_;
  vsip_stride         x_stride_;
  vsip_length         z_length_;
  vsip_length         y_length_;
  vsip_length         x_length_;
  map_type            map_;
};

template <typename S>
class Block<3, S, true>
{
  struct Derived_parameters 
  {
    vsip_offset offset;
    vsip_stride z_stride;
    vsip_stride y_stride;
    vsip_stride x_stride;
  };
public:
  static vsip::dimension_type const dim = 3;
  typedef typename S::value_type value_type;
  typedef value_type &           reference_type;
  typedef value_type const &     const_reference_type;
  typedef vsip::Local_map        map_type;

  typedef typename S::ptr_type ptr_type;
  typedef typename S::const_ptr_type const_ptr_type;

  Block(vsip_offset o,
        vsip_stride zs, vsip_length zl,
        vsip_stride ys, vsip_length yl,
        vsip_stride xs, vsip_length xl,
        S &storage,
        bool derived)
    : storage_(storage), offset_(o),
      z_stride_(zs), y_stride_(ys), x_stride_(xs),
      z_length_(zl), y_length_(yl), x_length_(xl),
      derived_params_(0)
  {
    if (derived)
    {
      derived_params_ = new Derived_parameters;
      derived_params_->offset = offset_;
      derived_params_->z_stride = z_stride_;
      derived_params_->y_stride = y_stride_;
      derived_params_->x_stride = x_stride_;
      // For interleaved adjust units.
      offset_ *= 2;
      z_stride_ *= 2;
      y_stride_ *= 2;
      x_stride_ *= 2;
    }
  }
  ~Block() { delete derived_params_;}

  void increment_count() const {}
  void decrement_count() const {}

  bool is_derived() const { return derived_params_;}

  vsip_offset offset() const
  { return derived_params_ ? derived_params_->offset : offset_;}
  void offset(vsip_offset o)
  {
    offset_ = o;
    if (derived_params_)
    {
      derived_params_->offset = offset_;
      offset_ *= 2;
    }
  }
  vsip_stride z_stride() const
  { return derived_params_ ? derived_params_->z_stride : z_stride_;}
  void z_stride(vsip_stride s)
  {
    z_stride_ = s;
    if (derived_params_)
    {
      derived_params_->z_stride = z_stride_;
      z_stride_ *= 2;
    }
  }
  vsip_stride y_stride()
    const { return derived_params_ ? derived_params_->y_stride : y_stride_;}
  void y_stride(vsip_stride s)
  {
    y_stride_ = s;
    if (derived_params_)
    {
      derived_params_->y_stride = y_stride_;
      y_stride_ *= 2;
    }
  }
  vsip_stride x_stride() const
  { return derived_params_ ? derived_params_->x_stride : x_stride_;}
  void x_stride(vsip_stride s)
  {
    x_stride_ = s;
    if (derived_params_)
    {
      derived_params_->x_stride = x_stride_;
      x_stride_ *= 2;
    }
  }
  vsip_length z_length() const { return z_length_;}
  void z_length(vsip_length l) { z_length_ = l;}
  vsip_length y_length() const { return y_length_;}
  void y_length(vsip_length l) { y_length_ = l;}
  vsip_length x_length() const { return x_length_;}
  void x_length(vsip_length l) { x_length_ = l;}

  vsip::length_type size() const { return z_length_*y_length_*x_length_;}
  vsip::length_type size(vsip::dimension_type D, vsip::dimension_type d) const
  {
    if (D == 1) return size();
    else return d == 0 ? z_length_ : d == 1 ? y_length_ : x_length_;
  }
  map_type const &map() const { return map_;}

  value_type get(vsip::index_type h, vsip::index_type i, vsip::index_type j) const 
  { return storage_.get(offset_ + h * z_stride_ + i * y_stride_ + j * x_stride_);}
  void put(vsip::index_type h, vsip::index_type i, vsip::index_type j, value_type v)
  { storage_.put(offset_ + h * z_stride_ + i * y_stride_ + j * x_stride_, v);}
  void put(vsip::index_type i, value_type v) 
  {
    vsip::index_type planes = y_length() * x_length();
    vsip::index_type z = i / planes;
    vsip::index_type y = (i - z * planes) % x_length();
    vsip::index_type x = i - z * planes - y * x_length();
    put(z, y, x, v);
  }

  // support direct data access protocol
  ptr_type ptr() { return impl::offset(storage_.ptr(), offset_);}
  const_ptr_type ptr() const { return impl::offset(storage_.ptr(), offset_);}
  vsip::stride_type stride(vsip::dimension_type D, vsip::dimension_type d) const
  { return D == 1 ? z_stride_ : d == 0 ? z_stride_ : d == 1 ? y_stride_ : x_stride_;}

private:
  S &                 storage_;
  vsip_offset         offset_;
  vsip_stride         z_stride_;
  vsip_stride         y_stride_;
  vsip_stride         x_stride_;
  vsip_length         z_length_;
  vsip_length         y_length_;
  vsip_length         x_length_;
  Derived_parameters *derived_params_;
  map_type            map_;
};

} // namespace vsip_csl::cvsip
} // namespace vsip_csl

namespace vsip
{
template <dimension_type D, typename S>
struct get_block_layout<vsip_csl::cvsip::Block<D, S> >
{
  static dimension_type const dim = D;

  typedef tuple<0, 1, 2>           order_type;
  static pack_type const packing = any_packing;
  static storage_format_type const storage_format = S::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <dimension_type D, typename S>
struct supports_dda<vsip_csl::cvsip::Block<D, S> >
{ static bool const value = true;};

} // namespace vsip

#endif

/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef vsip_core_stored_block_hpp_
#define vsip_core_stored_block_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/refcount.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/dense_storage.hpp>
#include <vsip/core/user_storage.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/parallel/local_map.hpp>
#include <stdexcept>

namespace vsip
{
namespace impl
{

/// Stored_block is a block-like type that stores its data
/// in (local) storage.
/// The default storage is Allocated_storage.
template <typename T, typename L>
class Stored_block : public Ref_count<Stored_block<T, L> >
{
  enum private_type {};
public:
  typedef T value_type;
  typedef T &reference_type;
  typedef T const &const_reference_type;

  typedef typename Complex_value_type<T, private_type>::type uT;

  typedef L layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef typename L::order_type order_type;

  typedef Allocated_storage<typename L::complex_type, T> storage_type;
  typedef typename storage_type::type data_type;
  typedef typename storage_type::const_type const_data_type;

  typedef Local_map map_type;

  static dimension_type const dim = L::dim;

  Stored_block(Domain<dim> const &dom, map_type const &map = map_type())
    : layout_(dom),
      storage_(layout_.total_size()),
      map_(map)
  {}

  Stored_block(Domain<dim> const &dom, T value, map_type const &map = map_type())
    : layout_(dom),
      storage_(layout_.total_size(), value),
      map_(map)
  {}

  ~Stored_block() { storage_.deallocate(layout_.total_size());}

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
    assert((block_dim == 1 || block_dim == dim) && (d < block_dim));

    if (block_dim == 1) return size();
    else return layout_.size(d);
  }

  value_type get(index_type idx) const VSIP_NOTHROW
  {
    assert(idx < size());
    return storage_.get(idx);
  }
  value_type get(index_type idx0, index_type idx1) const VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1));
    return storage_.get(layout_.index(idx0, idx1));
  }
  value_type get(index_type idx0, index_type idx1, index_type idx2) const VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1) &&
	   idx2 < layout_.size(2));
    return storage_.get(layout_.index(idx0, idx1, idx2));
  }

  void put(index_type idx, T val) VSIP_NOTHROW
  {
    assert(idx < size());
    storage_.put(idx, val);
  }
  void put(index_type idx0, index_type idx1, T val) VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1));
    storage_.put(layout_.index(idx0, idx1), val);
  }
  void put(index_type idx0, index_type idx1, index_type idx2, T val)
    VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1) &&
	   idx2 < layout_.size(2));
    storage_.put(layout_.index(idx0, idx1, idx2), val);
  }

  data_type impl_data() { return storage_.data();}
  const_data_type impl_data() const { return storage_.data();}
  stride_type impl_stride(dimension_type block_dim, dimension_type d) const
  {
    assert(block_dim == 1 || block_dim == dim);
    assert(d < dim);
    if (block_dim == 1) return 1;
    else return layout_.stride(d);
  }

private:
  applied_layout_type layout_;
  storage_type storage_;
  map_type map_;
};

/// Stored_block specialization that stores its
/// data in Dense_storage.
template <dimension_type D, typename T, typename O, typename F>
class Stored_block<T, Layout<D, O, Stride_unit_dense, F> >
  : public Ref_count<Stored_block<T, Layout<D, O, Stride_unit_dense, F> > >
{
  enum private_type {};
public:
  typedef T value_type;
  typedef T &reference_type;
  typedef T const &const_reference_type;
  typedef typename Complex_value_type<T, private_type>::type uT;

  typedef F complex_type;

  typedef Layout<D, O, Stride_unit_dense, F> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef O order_type;

  typedef Dense_storage<F, T> storage_type;
  typedef typename storage_type::type data_type;
  typedef typename storage_type::const_type const_data_type;

  typedef Local_map map_type;

  static dimension_type const dim = D;

  Stored_block(Domain<dim> const &dom, map_type const &map = map_type())
    : layout_(dom),
      storage_(map.impl_pool(), layout_.total_size()),
      map_(map),
      admitted_(true)
  {
    map_.impl_apply(dom);
  }

  Stored_block(Domain<dim> const &dom, T value, map_type const &map = map_type())
    : layout_(dom),
      storage_(map.impl_pool(), layout_.total_size(), value),
      map_(map),
      admitted_(true)
  {
    map_.impl_apply(dom);
  }

  /// User-storage constructor.
  ///
  /// If user_data is compatible with the storage_ type
  ///   (i.e. user_storage is array_format or interleaved_format and
  ///   storage_ is Cmplx_inter_fmt, or
  ///   user_storage is split_format and storage_ is Cmplx_split_fmt)
  /// then storage_ will use the user provided storage,
  /// else storage_ will allocate its own memory of the same size.
  ///
  /// On admit and release, storage_.is_alloc() is used to determine if
  /// storage_ and user_data_ are the same (in which case no copy is
  /// necessary to update), or different (copy necessary).
  Stored_block(Domain<dim> const &dom,
	       T *const pointer,
	       map_type const &map = map_type())
    : layout_(dom),
      user_data_(array_format, pointer),
      storage_(map.impl_pool(), layout_.total_size(),
               user_data_.as_storage(complex_type())),
      map_(map),
      admitted_(false)
  {
  }

  Stored_block(Domain<dim> const &dom,
	       uT *const pointer,
	       map_type const &map = map_type())
    : layout_(dom),
      user_data_(interleaved_format, pointer, 0),
      storage_(map.impl_pool(), layout_.total_size(),
	       user_data_.as_storage(complex_type())),
      map_(map),
      admitted_(false)
  {
  }

  Stored_block(Domain<D> const &dom,
	       uT *const real_pointer,
	       uT *const imag_pointer,
	       map_type const &map = map_type())
    : layout_(dom),
      user_data_(imag_pointer ? split_format : interleaved_format,
		 real_pointer, imag_pointer),
      storage_(map.impl_pool(), layout_.total_size(),
	       user_data_.as_storage(complex_type())),
      map_(map),
      admitted_(false)
  {
  }

  ~Stored_block() { storage_.deallocate(map_.impl_pool(), layout_.total_size());}

  /// Rebind user-storage to a new array.
  ///
  /// Requires
  ///   THIS must be a block with array_format user storage that is
  ///      currently released.
  void rebind(T* pointer)
  {
    assert(!admitted() && user_storage() == array_format);
    user_data_.rebind(pointer);
    storage_.rebind(map_.impl_pool(), layout_.total_size(),
		    user_data_.as_storage(complex_type()));
  }

  /// Rebind user-storage to a new interleaved array.
  ///
  /// Requires
  ///   THIS must be a block with interleaved_format or split_format
  ///      user storage that is currently released.
  ///
  /// Note:
  ///   When changing user storage from INTERLEAVED to ARRAY and
  ///    vice-versa, a rebind() will allocate/deallocate memory.
  void rebind(uT* pointer)
  {
    assert(!admitted() &&
	   (user_storage() == split_format ||
	    user_storage() == interleaved_format));
    user_data_.rebind(pointer);
    storage_.rebind(map_.impl_pool(), layout_.total_size(),
		    user_data_.as_storage(complex_type()));
  }

  /// Rebind user-storage to new split arrays.
  ///
  /// Requires:
  ///   THIS must be a block with interleaved_format or split_format
  ///      user storage that is currently released.
  ///
  /// Note:
  ///   When changing user storage from INTERLEAVED to ARRAY and
  ///    vice-versa, a rebind() will allocate/deallocate memory.
  void rebind(uT* real_pointer, uT* imag_pointer)
  {
    assert(!admitted() &&
	   (user_storage() == split_format ||
	    user_storage() == interleaved_format));
    user_data_.rebind(real_pointer, imag_pointer);
    storage_.rebind(map_.impl_pool(),
		    layout_.total_size(),
		    user_data_.as_storage(complex_type()));
  }

  // These three also resize the block.
  void rebind(T *pointer, Domain<D> const &dom)
  {
    assert(!admitted() && user_storage() == array_format);
    user_data_.rebind(pointer);
    storage_.deallocate(map_.impl_pool(), layout_.total_size());
    layout_ = applied_layout_type(dom);
    storage_.rebind(map_.impl_pool(), layout_.total_size(),
		    user_data_.as_storage(complex_type()));
  }

  void rebind(uT *pointer, Domain<D> const &dom)
  {
    assert(!admitted() &&
	   (user_storage() == split_format ||
	    user_storage() == interleaved_format));
    user_data_.rebind(pointer);
    storage_.deallocate(map_.impl_pool(), layout_.total_size());
    layout_ = applied_layout_type(dom);
    storage_.rebind(map_.impl_pool(), layout_.total_size(),
		    user_data_.as_storage(complex_type()));
  }
  void rebind(uT *real_pointer, uT *imag_pointer, Domain<D> const &dom)
  {
    assert(!admitted() &&
	   (user_storage() == split_format ||
	    user_storage() == interleaved_format));
    user_data_.rebind(real_pointer, imag_pointer);
    storage_.deallocate(map_.impl_pool(), layout_.total_size());
    layout_ = applied_layout_type(dom);
    storage_.rebind(map_.impl_pool(),
		    layout_.total_size(),
		    user_data_.as_storage(complex_type()));
  }

  void find(T *&pointer) VSIP_NOTHROW
  {
    assert(user_storage() == no_user_format ||
	   user_storage() == array_format);
    user_data_.find(pointer);
  }
  void find(uT *&pointer) VSIP_NOTHROW
  {
    assert(user_storage() == no_user_format ||
	   user_storage() == interleaved_format);
    user_data_.find(pointer);
  }
  void find(uT *&real_pointer, uT*& imag_pointer) VSIP_NOTHROW
  {
    assert(user_storage() == no_user_format ||
	   user_storage() == split_format);
    user_data_.find(real_pointer, imag_pointer);
  }

  void admit(bool update = true) VSIP_NOTHROW
  {
    if (!admitted_ && storage_.is_alloc() && update)
    {
      using namespace vsip::impl::profile;
      event<memory>("admit", size() * sizeof(value_type));
      for (index_type i = 0; i < size(); ++i)
	storage_.put(i, user_data_.get(i));
    }
    admitted_ = true;
  }
  /// Release user-storage from VSIPL++ control to user control.
  ///
  /// Note:
  ///  - It is not an error to release a block multiple times,
  ///    but it may signify an application programming error.
  void release(bool update = true) VSIP_NOTHROW
  {
    if (user_data_.format() == no_user_format)
      return;
    if (admitted_ && storage_.is_alloc() && update)
    {
      using namespace vsip::impl::profile;
      event<memory>("release", size() * sizeof(value_type));
      for (index_type i = 0; i < size(); ++i)
	this->user_data_.put(i, storage_.get(i));
    }
    admitted_ = false;
  }

  /// Release user-storage and return pointer (array format).
  ///
  /// Requires:
  ///   THIS to be a block with either array_format user storage,
  ///      or no_user_format user storage.
  void release(bool update, T*& pointer) VSIP_NOTHROW
  {
    assert(user_storage() == no_user_format ||
	   user_storage() == array_format);
    release(update);
    user_data_.find(pointer);
  }

  /// Release user-storeage and return pointer (interleaved format).
  ///
  /// Requires:
  ///   THIS to be a block with either interleaved_format user storage,
  ///      or no_user_format user storage.
  void release(bool update, uT*& pointer) VSIP_NOTHROW
  {
    assert(user_storage() == no_user_format ||
	   user_storage() == interleaved_format);
    release(update);
    user_data_.find(pointer);
  }

  /// Release user-storeage and return pointers (split format).
  ///
  /// Requires:
  ///   THIS to be a block with either split_format user storage,
  ///      or no_user_format user storage.
  void release(bool update, uT*& real_pointer, uT*& imag_pointer) VSIP_NOTHROW
  {
    assert(user_storage() == no_user_format ||
	   user_storage() == split_format);
    release(update);
    user_data_.find(real_pointer, imag_pointer);
  }

  user_storage_type user_storage() const VSIP_NOTHROW
  { return user_data_.format();}
  bool admitted() const VSIP_NOTHROW { return admitted_;}

  map_type const &map() const VSIP_NOTHROW { return map_;}

  length_type size() const VSIP_NOTHROW
  {
    length_type retval = layout_.size(0);
    for (dimension_type d = 1; d < D; ++d)
      retval *= layout_.size(d);
    return retval;
  }

  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  {
    assert((block_dim == 1 || block_dim == D) && (d < block_dim));

    if (block_dim == 1) return size();
    else return layout_.size(d);
  }

  value_type get(index_type idx) const VSIP_NOTHROW
  {
    assert(idx < size());
    return storage_.get(idx);
  }
  value_type get(index_type idx0, index_type idx1) const VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1));
    return storage_.get(layout_.index(idx0, idx1));
  }
  value_type get(index_type idx0, index_type idx1, index_type idx2) const VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1) &&
	   idx2 < layout_.size(2));
    return storage_.get(layout_.index(idx0, idx1, idx2));
  }

  void put(index_type idx, T val) VSIP_NOTHROW
  {
    assert(idx < size());
    storage_.put(idx, val);
  }
  void put(index_type idx0, index_type idx1, T val) VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1));
    storage_.put(layout_.index(idx0, idx1), val);
  }
  void put(index_type idx0, index_type idx1, index_type idx2, T val)
    VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1) &&
	   idx2 < layout_.size(2));
    storage_.put(layout_.index(idx0, idx1, idx2), val);
  }

  reference_type impl_ref(index_type idx) VSIP_NOTHROW
  {
    assert(idx < size());
    return storage_.ref(idx);
  }
  const_reference_type impl_ref(index_type idx) const VSIP_NOTHROW
  {
    assert(idx < size());
    return storage_.ref(idx);
  }
  reference_type impl_ref(Index<dim> const &idx) VSIP_NOTHROW
  {
    return storage_.ref(layout_.index(idx));
  }
  const_reference_type impl_ref(Index<dim> const &idx) const VSIP_NOTHROW
  {
    return storage_.ref(layout_.index(idx));
  }

  data_type impl_data() { return storage_.data();}
  const_data_type impl_data() const { return storage_.data();}
  stride_type impl_stride(dimension_type block_dim, dimension_type d) const
  {
    assert(block_dim == 1 || block_dim == D);
    assert(d < D);
    if (block_dim == 1) return 1;
    else return layout_.stride(d);
  }

private:
  applied_layout_type layout_;
  User_storage<T> user_data_;
  storage_type storage_;
  map_type map_;
  bool admitted_;
};

} // namespace vsip::impl
} // namespace vsip

#endif

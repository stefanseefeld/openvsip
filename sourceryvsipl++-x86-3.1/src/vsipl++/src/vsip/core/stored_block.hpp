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

template <typename T, typename L>
struct choose_storage
{
  typedef typename conditional<L::packing == dense,
			       Dense_storage<L::storage_format, T>,
			       Allocated_storage<L::storage_format, T> >::type type;
};

/// Stored_block is a block-like type that stores its data
/// in (local) storage.
/// The default storage is Allocated_storage.
template <typename T, typename L,
	  typename S = typename choose_storage<T, L>::type>
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

  typedef S storage_type;
  typedef typename storage_type::type ptr_type;
  typedef typename storage_type::const_type const_ptr_type;

  typedef Local_map map_type;

  static dimension_type const dim = L::dim;
  static storage_format_type const storage_format = L::storage_format;

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

  ptr_type ptr() { return storage_.ptr();}
  const_ptr_type ptr() const { return storage_.ptr();}
  stride_type stride(dimension_type block_dim, dimension_type d) const
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
template <dimension_type D, typename T, typename O, storage_format_type F, typename S>
class Stored_block<T, Layout<D, O, dense, F>, S>
  : public Ref_count<Stored_block<T, Layout<D, O, dense, F> > >
{
  enum private_type {};
public:
  typedef T value_type;
  typedef T &reference_type;
  typedef T const &const_reference_type;
  typedef typename Complex_value_type<T, private_type>::type uT;

  typedef Layout<D, O, dense, F> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef O order_type;

  typedef S storage_type;
  typedef typename storage_type::type ptr_type;
  typedef typename storage_type::const_type const_ptr_type;

  typedef Local_map map_type;

  static dimension_type const dim = D;
  static storage_format_type const storage_format = F;

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
  /// If user_data is compatible with the `storage_` type
  /// (i.e. user_storage is array_format or interleaved_format and
  /// `storage_` is interleaved_complex, or
  /// user_storage is split_format and `storage_` is split_complex)
  /// then `storage_` will use the user provided storage,
  /// else `storage_` will allocate its own memory of the same size.
  ///
  /// On admit and release, `storage_.is_alloc()` is used to determine if
  /// `storage_` and `user_data_` are the same (in which case no copy is
  /// necessary to update), or different (copy necessary).
  Stored_block(Domain<dim> const &dom,
	       T *const pointer,
	       map_type const &map = map_type())
    : layout_(dom),
      user_data_(array_format, pointer),
      storage_(map.impl_pool(), layout_.total_size(),
	       user_data_.template as<storage_format>()),
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
	       user_data_.template as<storage_format>()),
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
	       user_data_.template as<storage_format>()),
      map_(map),
      admitted_(false)
  {
  }

  Stored_block(Domain<D> const &dom,
	       std::pair<uT*,uT*> pointer,
	       map_type const &map = map_type())
    : layout_(dom),
      user_data_(pointer.second ? split_format : interleaved_format,
		 pointer.first, pointer.second),
      storage_(map.impl_pool(), layout_.total_size(),
	       user_data_.template as<storage_format>()),
      map_(map),
      admitted_(false)
  {
  }

  ~Stored_block() { storage_.deallocate(map_.impl_pool(), layout_.total_size());}

  /// Rebind user-storage to a new array.
  ///
  /// Requires
  ///   The block must be released.
  void rebind(T* pointer)
  {
    assert(!admitted());
    user_data_.rebind(pointer);
    storage_.rebind(map_.impl_pool(), layout_.total_size(),
		    user_data_.template as<storage_format>());
  }

  /// Rebind user-storage to a new interleaved array.
  ///
  /// Requires
  ///   The block must be released.
  ///
  /// Note:
  ///   When changing user storage from INTERLEAVED to ARRAY and
  ///    vice-versa, a rebind() will allocate/deallocate memory.
  void rebind(uT* pointer)
  {
    assert(!admitted());
    user_data_.rebind(pointer);
    storage_.rebind(map_.impl_pool(), layout_.total_size(),
		    user_data_.template as<storage_format>());
  }

  /// Rebind user-storage to new split arrays.
  ///
  /// Requires:
  ///   The block must be released.
  ///
  /// Note:
  ///   When changing user storage from INTERLEAVED to ARRAY and
  ///    vice-versa, a rebind() will allocate/deallocate memory.
  void rebind(uT* real_pointer, uT* imag_pointer)
  {
    assert(!admitted());
    user_data_.rebind(real_pointer, imag_pointer);
    storage_.rebind(map_.impl_pool(),
		    layout_.total_size(),
		    user_data_.template as<storage_format>());
  }
  void rebind(std::pair<uT*,uT*> pointer) { rebind(pointer.first, pointer.second);}

  // These variants also resize the block.
  void rebind(T *pointer, Domain<D> const &dom)
  {
    assert(!admitted());
    user_data_.rebind(pointer);
    storage_.deallocate(map_.impl_pool(), layout_.total_size());
    layout_ = applied_layout_type(dom);
    storage_.rebind(map_.impl_pool(), layout_.total_size(),
		    user_data_.template as<storage_format>());
  }

  void rebind(uT *pointer, Domain<D> const &dom)
  {
    assert(!admitted());
    user_data_.rebind(pointer);
    storage_.deallocate(map_.impl_pool(), layout_.total_size());
    layout_ = applied_layout_type(dom);
    storage_.rebind(map_.impl_pool(), layout_.total_size(),
		    user_data_.template as<storage_format>());
  }
  void rebind(uT *real_pointer, uT *imag_pointer, Domain<D> const &dom)
  {
    assert(!admitted());
    user_data_.rebind(real_pointer, imag_pointer);
    storage_.deallocate(map_.impl_pool(), layout_.total_size());
    layout_ = applied_layout_type(dom);
    storage_.rebind(map_.impl_pool(),
		    layout_.total_size(),
		    user_data_.template as<storage_format>());
  }
  void rebind(std::pair<uT*,uT*> pointer, Domain<D> const &dom)
  { rebind(pointer.first, pointer.second, dom);}

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
  void find(std::pair<uT*,uT*> &pointer) VSIP_NOTHROW { find(pointer.first, pointer.second);}

  void admit(bool update = true) VSIP_NOTHROW
  {
    if (!admitted_ && storage_.is_alloc() && update)
    {
      namespace p = vsip::impl::profile;
      p::Scope<p::copy>("admit", size() * sizeof(value_type));
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
      namespace p = vsip::impl::profile;
      p::Scope<p::copy>("release", size() * sizeof(value_type));
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

  /// Release user-storage and return pointers (split format).
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

  void release(bool update, std::pair<uT*,uT*> &pointer) VSIP_NOTHROW
  { release(update, pointer.first, pointer.second);}

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

  ptr_type ptr() { return storage_.ptr();}
  const_ptr_type ptr() const { return storage_.ptr();}
  stride_type stride(dimension_type block_dim, dimension_type d) const
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

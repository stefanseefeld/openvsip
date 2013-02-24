/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef vsip_opt_cuda_stored_block_hpp_
#define vsip_opt_cuda_stored_block_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/refcount.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/dense_storage.hpp>
#include <vsip/core/user_storage.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/parallel/local_map.hpp>
#include <vsip/opt/cuda/device_storage.hpp>
#include <stdexcept>

namespace vsip
{
namespace impl
{
namespace cuda
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
  enum storage_state_type
  {
    INVALID = 0, // unused
    HOST_VALID = 1,
    DEVICE_VALID = 2,
    VALID = HOST_VALID | DEVICE_VALID
  };

  typedef T value_type;
  typedef T &reference_type;
  typedef T const &const_reference_type;

  typedef typename Complex_value_type<T, private_type>::type uT;

  typedef L layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef typename L::order_type order_type;

  typedef S host_storage_type;
  typedef typename host_storage_type::type ptr_type;
  typedef typename host_storage_type::const_type const_ptr_type;

  typedef cuda::Device_storage<T, L> device_storage_type;

  typedef Local_map map_type;

  static dimension_type const dim = L::dim;
  static storage_format_type const storage_format = L::storage_format;

  Stored_block(Domain<dim> const &dom, map_type const &map = map_type())
    : layout_(dom),
      host_storage_(layout_.total_size()),
      device_storage_(layout_),
      storage_state_(HOST_VALID),
      map_(map)
  {}

  Stored_block(Domain<dim> const &dom, T value, map_type const &map = map_type())
    : layout_(dom),
      host_storage_(layout_.total_size(), value),
      device_storage_(layout_),
      storage_state_(HOST_VALID),
      map_(map)
  {}

  ~Stored_block() { host_storage_.deallocate(layout_.total_size());}

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
    sync_host();
    return host_storage_.get(idx);
  }
  value_type get(index_type idx0, index_type idx1) const VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1));
    sync_host();
    return host_storage_.get(layout_.index(idx0, idx1));
  }
  value_type get(index_type idx0, index_type idx1, index_type idx2) const VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1) &&
	   idx2 < layout_.size(2));
    sync_host();
    return host_storage_.get(layout_.index(idx0, idx1, idx2));
  }

  void put(index_type idx, T val) VSIP_NOTHROW
  {
    assert(idx < size());
    sync_host();
    host_storage_.put(idx, val);
    invalidate_device();
  }
  void put(index_type idx0, index_type idx1, T val) VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1));
    sync_host();
    host_storage_.put(layout_.index(idx0, idx1), val);
    invalidate_device();
  }
  void put(index_type idx0, index_type idx1, index_type idx2, T val)
    VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1) &&
	   idx2 < layout_.size(2));
    sync_host();
    host_storage_.put(layout_.index(idx0, idx1, idx2), val);
    invalidate_device();
  }

  ptr_type ptr()
  {
    sync_host();
    invalidate_device(); // data may change
    return host_storage_.ptr();
  }
  const_ptr_type ptr() const
  {
    sync_host();
    return host_storage_.ptr();
  }
  ptr_type device_ptr()
  {
    // This will copy data to the device if and only if necessary.
    sync_device();
    // Because device data may change, we mark the host-side buffer
    // appropriately.
    invalidate_host();
    return device_storage_.ptr();
  }
  const_ptr_type device_ptr() const
  {
    // This will copy data to the device if and only if necessary.
    sync_device();
    return device_storage_.ptr();
  }
  stride_type stride(dimension_type block_dim, dimension_type d) const
  {
    assert(block_dim == 1 || block_dim == dim);
    assert(d < dim);
    if (block_dim == 1) return 1;
    else return layout_.stride(d);
  }
  stride_type device_stride(dimension_type block_dim, dimension_type d) const
  {
    assert(block_dim == 1 || block_dim == dim);
    assert(d < dim);
    if (block_dim == 1) return 1;
    else return device_storage_.stride(d);
  }

  storage_state_type impl_storage_state() const { return storage_state_;}

  // Right now, either device or host storage is valid.
  // Marking one as invalid implies the other is or becomes valid.
  void invalidate_device() { storage_state_ = HOST_VALID;}
  void invalidate_host() { storage_state_ = DEVICE_VALID;}

private:
  void sync_host() const
  {
    if (storage_state_ == DEVICE_VALID)
    {
      device_storage_.to_host(host_storage_.ptr());
      storage_state_ = VALID;
    }
  }
  void sync_device() const
  {
    if (storage_state_ == HOST_VALID)
    {
      device_storage_.from_host(host_storage_.ptr());
      storage_state_ = VALID;
    }
  }

  applied_layout_type layout_;
  mutable host_storage_type host_storage_;
  mutable device_storage_type device_storage_;
  mutable storage_state_type storage_state_;
  map_type map_;
};

/// Stored_block specialization that stores its
/// data in Dense_storage.
template <dimension_type D, typename T, typename O, storage_format_type C, typename S>
class Stored_block<T, Layout<D, O, dense, C>, S>
  : public Ref_count<Stored_block<T, Layout<D, O, dense, C> > >
{
  enum private_type {};

public:
  enum storage_state_type
  {
    INVALID = 0, // unused
    HOST_VALID = 1,
    DEVICE_VALID = 2,
    VALID = HOST_VALID | DEVICE_VALID
  };

  typedef T value_type;
  typedef T &reference_type;
  typedef T const &const_reference_type;
  typedef typename Complex_value_type<T, private_type>::type uT;

  typedef Layout<D, O, dense, C> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef O order_type;

  typedef S host_storage_type;
  typedef typename host_storage_type::type ptr_type;
  typedef typename host_storage_type::const_type const_ptr_type;

  typedef cuda::Device_storage<T, layout_type> device_storage_type;

  typedef Local_map map_type;

  static dimension_type const dim = D;
  static storage_format_type const storage_format = C;

  Stored_block(Domain<dim> const &dom, map_type const &map = map_type())
    : layout_(dom),
      host_storage_(map.impl_pool(), layout_.total_size()),
      device_storage_(layout_),
      storage_state_(HOST_VALID),
      map_(map),
      admitted_(true)
  {
    map_.impl_apply(dom);
  }

  Stored_block(Domain<dim> const &dom, T value, map_type const &map = map_type())
    : layout_(dom),
      host_storage_(map.impl_pool(), layout_.total_size(), value),
      device_storage_(layout_),
      storage_state_(HOST_VALID),
      map_(map),
      admitted_(true)
  {
    map_.impl_apply(dom);
  }

  /// User-storage constructor.
  ///
  /// If user_data is compatible with the storage_ type
  ///   (i.e. user_storage is array_format or interleaved_format and
  ///   storage_ is interleaved_complex, or
  ///   user_storage is split_format and storage_ is split_complex)
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
      host_storage_(map.impl_pool(), layout_.total_size(), user_data_.template as<C>()),
      device_storage_(layout_),
      storage_state_(HOST_VALID),
      map_(map),
      admitted_(false)
  {
  }

  Stored_block(Domain<dim> const &dom,
	       uT *const pointer,
	       map_type const &map = map_type())
    : layout_(dom),
      user_data_(interleaved_format, pointer, 0),
      host_storage_(map.impl_pool(), layout_.total_size(), user_data_.template as<C>()),
      device_storage_(layout_),
      storage_state_(HOST_VALID),
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
      host_storage_(map.impl_pool(), layout_.total_size(), user_data_.template as<C>()),
      device_storage_(layout_),
      storage_state_(HOST_VALID),
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
      host_storage_(map.impl_pool(), layout_.total_size(), user_data_.template as<C>()),
      device_storage_(layout_),
      storage_state_(HOST_VALID),
      map_(map),
      admitted_(false)
  {
  }

  ~Stored_block() { host_storage_.deallocate(map_.impl_pool(), layout_.total_size());}

  /// Rebind user-storage to a new array.
  ///
  /// Requires
  ///   The block must be released.
  void rebind(T* pointer)
  {
    assert(!admitted());
    user_data_.rebind(pointer);
    host_storage_.rebind(map_.impl_pool(), layout_.total_size(), user_data_.template as<C>());
    invalidate_device();
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
    host_storage_.rebind(map_.impl_pool(), layout_.total_size(), user_data_.template as<C>());
    invalidate_device();
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
    host_storage_.rebind(map_.impl_pool(), layout_.total_size(), user_data_.template as<C>());
    invalidate_device();
  }
  void rebind(std::pair<uT*,uT*> pointer) { rebind(pointer.first, pointer.second);}

  // These variants also resize the block.
  void rebind(T *pointer, Domain<D> const &dom)
  {
    assert(!admitted());
    user_data_.rebind(pointer);
    host_storage_.deallocate(map_.impl_pool(), layout_.total_size());
    layout_ = applied_layout_type(dom);
    host_storage_.rebind(map_.impl_pool(), layout_.total_size(), user_data_.template as<C>());
    device_storage_.resize(layout_.total_size());
    invalidate_device();
  }

  void rebind(uT *pointer, Domain<D> const &dom)
  {
    assert(!admitted());
    user_data_.rebind(pointer);
    host_storage_.deallocate(map_.impl_pool(), layout_.total_size());
    layout_ = applied_layout_type(dom);
    host_storage_.rebind(map_.impl_pool(), layout_.total_size(), user_data_.template as<C>());
    device_storage_.resize(layout_.total_size());
    invalidate_device();
  }
  void rebind(uT *real_pointer, uT *imag_pointer, Domain<D> const &dom)
  {
    assert(!admitted());
    user_data_.rebind(real_pointer, imag_pointer);
    host_storage_.deallocate(map_.impl_pool(), layout_.total_size());
    layout_ = applied_layout_type(dom);
    host_storage_.rebind(map_.impl_pool(), layout_.total_size(), user_data_.template as<C>());
    device_storage_.resize(layout_.total_size());
    invalidate_device();
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
    if (!admitted_ && host_storage_.is_alloc() && update)
    {
      namespace p = vsip::impl::profile;
      p::Scope<p::copy>("admit", size() * sizeof(value_type));
      for (index_type i = 0; i < size(); ++i)
	host_storage_.put(i, user_data_.get(i));
      invalidate_device();
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
    if (admitted_ && host_storage_.is_alloc() && update)
    {
      namespace p = vsip::impl::profile;
      p::Scope<p::copy>("release", size() * sizeof(value_type));
      sync_host();
      for (index_type i = 0; i < size(); ++i)
	this->user_data_.put(i, host_storage_.get(i));
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
    sync_host();
    return host_storage_.get(idx);
  }
  value_type get(index_type idx0, index_type idx1) const VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1));
    sync_host();
    return host_storage_.get(layout_.index(idx0, idx1));
  }
  value_type get(index_type idx0, index_type idx1, index_type idx2) const VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1) &&
	   idx2 < layout_.size(2));
    sync_host();
    return host_storage_.get(layout_.index(idx0, idx1, idx2));
  }

  void put(index_type idx, T val) VSIP_NOTHROW
  {
    assert(idx < size());
    sync_host();
    host_storage_.put(idx, val);
    invalidate_device();
  }
  void put(index_type idx0, index_type idx1, T val) VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1));
    sync_host();
    host_storage_.put(layout_.index(idx0, idx1), val);
    invalidate_device();
  }
  void put(index_type idx0, index_type idx1, index_type idx2, T val)
    VSIP_NOTHROW
  {
    assert(idx0 < layout_.size(0) && idx1 < layout_.size(1) &&
	   idx2 < layout_.size(2));
    sync_host();
    host_storage_.put(layout_.index(idx0, idx1, idx2), val);
    invalidate_device();
  }

  reference_type impl_ref(index_type idx) VSIP_NOTHROW
  {
    assert(idx < size());
    sync_host();
    invalidate_device(); // data may change.
    return host_storage_.ref(idx);
  }
  const_reference_type impl_ref(index_type idx) const VSIP_NOTHROW
  {
    assert(idx < size());
    sync_host();
    return host_storage_.ref(idx);
  }
  reference_type impl_ref(Index<dim> const &idx) VSIP_NOTHROW
  {
    sync_host();
    invalidate_device(); // data may change
    return host_storage_.ref(layout_.index(idx));
  }
  const_reference_type impl_ref(Index<dim> const &idx) const VSIP_NOTHROW
  {
    sync_host();
    return host_storage_.ref(layout_.index(idx));
  }

  ptr_type ptr()
  {
    sync_host();
    invalidate_device(); // data may change
    return host_storage_.ptr();
  }
  const_ptr_type ptr() const
  {
    sync_host();
    return host_storage_.ptr();
  }
  ptr_type device_ptr()
  {
    // This will copy data to the device if and only if necessary.
    sync_device();
    // Because device data may change, we mark the host-side buffer
    // appropriately.
    invalidate_host();
    return device_storage_.ptr();
  }
  const_ptr_type device_ptr() const
  {
    // This will copy data to the device if and only if necessary.
    sync_device();
    return device_storage_.ptr();
  }
  stride_type stride(dimension_type block_dim, dimension_type d) const
  {
    assert(block_dim == 1 || block_dim == D);
    assert(d < D);
    if (block_dim == 1) return 1;
    else return layout_.stride(d);
  }
  stride_type device_stride(dimension_type block_dim, dimension_type d) const
  {
    assert(block_dim == 1 || block_dim == D);
    assert(d < D);
    if (block_dim == 1) return 1;
    else return device_storage_.stride(d);
  }

  storage_state_type impl_storage_state() const { return storage_state_;}

  // Right now, either device or host storage is valid.
  // Marking one as invalid implies the other is or becomes valid.
  void invalidate_device() { storage_state_ = HOST_VALID;}
  void invalidate_host() { storage_state_ = DEVICE_VALID;}

private:
  void sync_host() const
  {
    if (storage_state_ == DEVICE_VALID)
    {
      device_storage_.to_host(host_storage_.ptr());
      storage_state_ = VALID;
    }
  }
  void sync_device() const
  {
    if (storage_state_ == HOST_VALID)
    {
      device_storage_.from_host(host_storage_.ptr());
      storage_state_ = VALID;
    }
  }

  applied_layout_type layout_;
  User_storage<T> user_data_;
  mutable host_storage_type host_storage_;
  mutable device_storage_type device_storage_;
  mutable storage_state_type storage_state_;
  map_type map_;
  bool admitted_;
};

/// Report whether the given block has internal device-storage.
template <typename Block>
struct has_device_storage { static bool const value = false;};

template <typename B>
struct has_device_storage<B const> : has_device_storage<B> {};

/// Report whether the given block's data resides on the device.
/// Note that this is potentially ambiguous, since a block's data
/// may be valid on host memory *and* device memory at the same time.
/// To disambiguate, device data takes precedence over host data, i.e
/// whenever the block's device storage is valid, we use that.
template <typename Block, bool HasDevice = has_device_storage<Block>::value>
struct is_device_data_valid 
{
  static bool check(Block const &) { return false;}
};

template <typename Block>
bool has_valid_device_ptr(Block const &b) 
{ return is_device_data_valid<Block>::check(b);}

template <typename B>
struct is_device_data_valid<B, true> 
{
  static bool check(B const &b) 
  { return b.impl_storage_state() == B::DEVICE_VALID;}
};

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif

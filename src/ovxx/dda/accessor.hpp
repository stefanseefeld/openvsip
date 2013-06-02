//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_dda_accessor_hpp_
#define ovxx_dda_accessor_hpp_

#include <ovxx/ct_assert.hpp>
#include <ovxx/storage.hpp>
#include <ovxx/domain_utils.hpp>
#include <ovxx/dda/block_copy.hpp>
#include <ovxx/view/traits.hpp>
#include <ovxx/parallel/assign_local.hpp>
#include <ovxx/adjust_layout.hpp>
#include <ovxx/strided.hpp>
#include <ovxx/layout.hpp>

namespace vsip
{
namespace dda
{
typedef unsigned sync_policy;

sync_policy const in = 0x01;  ///< Synchronize input block on DDA creation
sync_policy const out = 0x02; ///< Synchronize output block on DDA destruction
sync_policy const inout = in | out; ///< Synchronize inout block in both directions
sync_policy const copy = 0x04; ///< Force copy
sync_policy const device = 0x08; ///< Request device storage

} // namespace vsip::dda
} // namespace vsip

namespace ovxx
{
namespace dda
{
using namespace vsip::dda;

/// Determine if 'block' is compatible with
/// the given layout 'L'.
template <typename L, typename B>
bool
is_compatible(B const &block)
{
  typedef typename B::value_type value_type;
  static storage_format_type const storage_format 
    = get_block_layout<B>::storage_format;
  typedef typename get_block_layout<B>::order_type order_type;

  dimension_type const dim0 = order_type::impl_dim0;
  dimension_type const dim1 = order_type::impl_dim1;
  dimension_type const dim2 = order_type::impl_dim2;

  if (storage_format != L::storage_format ||
      !is_same<order_type, typename L::order_type>::value)
    return false;
  else if (L::packing == unit_stride)
  {
    if (L::dim == 1)
      return block.stride(1, 0) == 1;
    else if (L::dim == 2)
      return block.stride(2, dim1) == 1;
    else // L::dim == 3
      return block.stride(3, dim2) == 1;
  }
  else if (L::packing == dense)
  {
    if (L::dim == 1)
      return block.stride(1, 0) == 1;
    else if (L::dim == 2)
      return block.stride(2, dim1) == 1
	     && (block.stride(2, dim0) ==
		 static_cast<stride_type>(block.size(2, dim1))
		 || block.size(2, dim0) == 1);
    else // L::dim == 3
    {

      bool ok2 = (block.stride(3, dim2) == 1);
      bool ok1 = (block.stride(3, dim1) ==
		  static_cast<stride_type>(block.size(3, dim2)))
	|| (block.size(3, dim1) == 1 && block.size(3, dim0) == 1);
      bool ok0 = (block.stride(3, dim0) ==
		  static_cast<stride_type>(block.size(3, dim1) *
					   block.size(3, dim2)))
	|| block.size(3, dim0) == 1;

      return ok0 && ok1 && ok2;
    }
  }
  else if (is_packing_aligned<L::packing>::value)
  {
    unsigned alignment = is_packing_aligned<L::packing>::alignment;

    if (!is_aligned_to(block.ptr(), alignment)) return false;

    if (L::dim == 1)
      return block.stride(1, 0) == 1;
    else if (L::dim == 2)
      return block.stride(2, dim1) == 1 &&
	((block.stride(2, dim0) * sizeof(value_type)) % alignment == 0);
    else // L::dim == 3
      return 
	block.stride(3, dim2) == 1 &&
	(block.stride(3, dim1) * sizeof(value_type)) % alignment == 0 &&
	(block.stride(3, dim0) * sizeof(value_type)) % alignment == 0;
  }
  else // any_packing
  {
    OVXX_PRECONDITION(L::packing == any_packing);
    return true;
  }
}

template <typename B, typename L>
bool
is_compatible(B const &block, L const &layout)
{
  typedef typename B::value_type value_type;
  typedef typename get_block_layout<B>::type block_layout_type;

  dimension_type const dim = L::dim;

  block_layout_type block_layout;

  dimension_type const dim0 = layout_nth_dim(block_layout, 0);
  dimension_type const dim1 = layout_nth_dim(block_layout, 1);
  dimension_type const dim2 = layout_nth_dim(block_layout, 2);

  if (is_complex<value_type>::value &&
      // storage formats only need to match for split-complex;
      // array and interleaved-complex can be cast to each other
      (layout_storage_format(block_layout) == split_complex) != 
      (layout_storage_format(layout) == split_complex))
    return false;

  for (dimension_type d=0; d<dim; ++d)
    if (layout_nth_dim(block_layout, d) != layout_nth_dim(layout, d))
      return false;

  if (layout_packing(layout) == unit_stride)
  {
    if (dim == 1) return block.stride(1, 0) == 1;
    else if (dim == 2) return block.stride(2, dim1) == 1;
    else return block.stride(3, dim2) == 1;
  }
  else if (layout_packing(layout) == dense)
  {
    if (dim == 1) return block.stride(1, 0) == 1;
    else if (dim == 2)
      return block.stride(2, dim1) == 1
	&& (block.stride(2, dim0) == static_cast<stride_type>(block.size(2, dim1))
	    || block.size(2, dim0) == 1);
    else // dim == 3
    {
      bool ok2 = (block.stride(3, dim2) == 1);
      bool ok1 = (block.stride(3, dim1) ==
		  static_cast<stride_type>(block.size(3, dim2)))
	|| (block.size(3, dim1) == 1 && block.size(3, dim0) == 1);
      bool ok0 = (block.stride(3, dim0) ==
		  static_cast<stride_type>(block.size(3, dim1) *
					   block.size(3, dim2)))
	|| block.size(3, dim0) == 1;

      return ok0 && ok1 && ok2;
    }
  }
  else if (is_aligned(layout_packing(layout)))
  {
    unsigned alignment = layout_alignment(layout);

    if (!is_aligned_to(block.ptr(), alignment))
      return false;

    if (dim == 1)
      return block.stride(1, 0) == 1;
    else if (dim == 2)
      return block.stride(2, dim1) == 1 &&
	((block.stride(2, dim0) * sizeof(value_type)) % alignment == 0);
    else // L::dim == 3
      return 
	block.stride(3, dim2) == 1 &&
	(block.stride(3, dim1) * sizeof(value_type)) % alignment == 0 &&
	(block.stride(3, dim0) * sizeof(value_type)) % alignment == 0;
  }
  else // any_packing
  {
    OVXX_PRECONDITION(layout_packing(layout) == any_packing);
    return true;
  }
}

enum access_kind { direct_access,       ///< direct access
		   copy_access,         ///< copy access
		   maybe_direct_access, ///< may be direct access
		   local_access,        ///< direct access via local block
		   remote_access};      ///< remote-copy access

/// Associate a cost with each access-type.
/// This is an upper compile-time estimate.
/// The actual cost may be smaller (for 'maybe_direct', notably).
template <access_kind K> struct cost { static int const value = 10;};
template <> struct cost<direct_access> { static int const value = 0;};
template <> struct cost<copy_access> { static int const value = 2;};
template <> struct cost<maybe_direct_access> { static int const value = 2;};
template <> struct cost<remote_access> { static int const value = 10;};

/// Return access type for a given block and desired layout.
template <typename Block, typename L, sync_policy S, typename E = void>
struct get_block_access
{
  static dimension_type const dim = L::dim;
  typedef typename ovxx::remove_const<Block>::type block_type;
  typedef typename block_type::map_type map_type;

  static bool const is_local = is_same<Local_map, map_type>::value;
  static bool const is_local_compatible =
    is_layout_compatible<block_type, L>::value &&
    is_same<Replicated_map<dim>, map_type>::value;
  static bool const maybe_supports_direct = !(S&copy) &&
    supports_dda<block_type>::value && 
    maybe_layout_compatible<block_type, L>::value;
  static bool const supports_direct =
    maybe_supports_direct &&
    is_layout_compatible<block_type, L>::value;

  static access_kind const value =
    is_local ? (supports_direct ? direct_access :
		maybe_supports_direct ? maybe_direct_access : copy_access) :
    is_local_compatible ? local_access : remote_access;
};

// The semantics of accessing a block with runtime layout
// requirements are different: much of the decision is
// done at runtime. The only thing we do know is the block's
// own accessibility using its own layout.
template <typename B, dimension_type D, sync_policy S, typename E>
struct get_block_access<B, Rt_layout<D>, S, E>
{
  static access_kind const value =
    supports_dda<B>::value ? maybe_direct_access : copy_access;
};

/// Low-level data access class.
///
/// Template parameters:
///
///   :B: block that supports the data access interface indicated by `K`
///   :L: requested layout
///   :S: synchronization policy
template <typename B,
	  typename L,
	  sync_policy S,
	  access_kind K = get_block_access<B, L, S>::value,
	  typename E = void>
class Accessor;

/// Specialization for direct data access.
template <typename B, typename L, sync_policy S>
class Accessor<B, L, S, direct_access,
	       typename enable_if<!(S&device)>::type>
{
public:
  static access_kind const access = direct_access;
  typedef typename adjust_layout<L, typename get_block_layout<B>::type>::type
    layout_type;
  static dimension_type const dim = layout_type::dim;

  typedef typename B::value_type value_type;
  typedef typename layout_type::order_type order_type;
  static pack_type const packing = get_block_layout<B>::type::packing;
  static storage_format_type const storage_format = layout_type::storage_format;

  typedef storage_traits<value_type, storage_format> storage;
  typedef typename storage::ptr_type non_const_ptr_type;
  typedef typename storage::const_ptr_type const_ptr_type;
  typedef typename conditional<is_modifiable_block<B>::value,
			       non_const_ptr_type,
			       const_ptr_type>::type ptr_type;

  static int const ct_cost = cost<direct_access>::value;

  static int cost(B const &) { return ct_cost;}
  static length_type required_buffer_size(B const &) { return 0;}

  Accessor(B &b, non_const_ptr_type = non_const_ptr_type()) : block_(b) {}
  ~Accessor() {}

  void sync_in() {}
  void sync_out() {}

  B &block() { return block_;}
  B const &block() const { return block_;}

  int cost() const { return ct_cost;}

  ptr_type ptr() const { return pointer_cast<ptr_type>(block_.ptr());}
  stride_type stride(dimension_type d) const { return block_.stride(dim, d);}
  length_type size(dimension_type d) const { return block_.size(dim, d);}
  length_type size() const { return block_.size();}
  length_type storage_size() const { return block_.size();}

private:
  B &block_;
};

/// Specialization for distributed blocks with compatible layout.
/// Operates on local subblock of `B`.
template <typename B, typename L, sync_policy S>
class Accessor<B, L, S, local_access,
	       typename enable_if<!(S&device)>::type>
{
  typedef typename remove_const<B>::type non_const_block_type;
  typedef typename add_const<B>::type const_block_type;
  typedef typename conditional<is_const<B>::value,
    typename distributed_local_block<non_const_block_type>::type const,
    typename distributed_local_block<non_const_block_type>::type>::type
  local_block_type;
  typedef Accessor<local_block_type, L, S> local_access_type;

public:
  static access_kind const access = local_access;
  typedef typename local_access_type::layout_type layout_type;
  static dimension_type const dim = L::dim;

  typedef typename B::value_type value_type;
  typedef typename layout_type::order_type order_type;
  static pack_type const packing = layout_type::packing;
  static storage_format_type const storage_format = layout_type::storage_format;

  typedef host_storage<value_type, storage_format> storage_type;
  typedef typename storage_type::ptr_type non_const_ptr_type;
  typedef typename storage_type::const_ptr_type const_ptr_type;
  typedef typename conditional<is_modifiable_block<B>::value,
			       non_const_ptr_type,
			       const_ptr_type>::type ptr_type;

  static int   const ct_cost = local_access_type::ct_cost;

  static int cost(B const &) { return ct_cost;}
  static length_type required_buffer_size(B const &b)
  { return local_access_type::required_buffer_size(b);}

  Accessor(B &b, non_const_ptr_type buffer = non_const_ptr_type())
    : local_access_(local_cast(b), buffer)
  {}
  ~Accessor() {}

  void sync_in() { local_access_.sync_in();}
  void sync_out() { local_access_.sync_out();}

  int cost() const { return local_access_.cost();}

  ptr_type ptr() const { return local_access_.ptr();}
  stride_type  stride(dimension_type d) const 
  { return local_access_.stride(d);}
  length_type  size(dimension_type d) const 
  { return local_access_.size(d);}
  length_type  size() const { return local_access_.size();}
  length_type storage_size() const { return local_access_.storage_size();}

private:
  static local_block_type &local_cast(B &b)
  { return get_local_block(const_cast<non_const_block_type &>(b));}

  local_access_type local_access_;
};

/// Specialization for copied direct data access.
template <typename B, typename L, sync_policy S>
class Accessor<B, L, S, copy_access,
	       typename enable_if<!(S&device)>::type>
{
  typedef typename get_block_layout<B>::type block_layout_type;
  static pack_type const packing = 
    static_cast<pack_type>(L::packing == unit_stride || L::packing == any_packing ? dense : L::packing);
  static storage_format_type const storage_format =
    adjust_storage_format<L::storage_format, block_layout_type::storage_format>::value;
  typedef typename adjust_type<typename L::order_type, typename block_layout_type::order_type>::type order_type;
  
public:
  static access_kind const access = copy_access;
  static dimension_type const dim = L::dim;
  typedef Layout<dim, order_type, packing, storage_format> layout_type;

  typedef typename B::value_type value_type;

  typedef host_storage<value_type, storage_format> storage_type;
  typedef typename storage_type::ptr_type non_const_ptr_type;
  typedef typename storage_type::const_ptr_type const_ptr_type;
  typedef typename conditional<is_modifiable_block<B>::value,
			       non_const_ptr_type,
			       const_ptr_type>::type ptr_type;

  static int   const ct_cost = cost<copy_access>::value;

  static int    cost(B const &) { return ct_cost;}
  static length_type required_buffer_size(B const &b)
  {
    Applied_layout<layout_type> layout(extent<dim>(b));
    return layout.total_size();
  }

  Accessor(B &block, non_const_ptr_type buffer = non_const_ptr_type())
    : block_(block),
      layout_(extent<dim>(block_)),
      storage_(layout_.total_size(), buffer),
      dirty_(false)
  {
    sync_in();
  }
  ~Accessor()
  {
    sync_out();
  }

  void sync_in() 
  {
    if (S&in) block_copy(block_, storage_.ptr(), layout_);
  }
  void sync_out()
  {
    if (S&out && dirty_) block_copy(storage_.ptr(), layout_, block_);
  }

  Applied_layout<layout_type> const &layout() { return layout_;}

  int cost() const { return ct_cost;}

  ptr_type ptr() { dirty_ = true; return storage_.ptr();}
  const_ptr_type ptr() const { return storage_.ptr();}
  stride_type stride(dimension_type d) const { return layout_.stride(d);}
  length_type size(dimension_type d) const { return block_.size(dim, d);}
  length_type size() const { return block_.size();}
  length_type storage_size() const { return layout_.total_size();}

private:
  B &block_;
  Applied_layout<layout_type> layout_;
  storage_type storage_;
  bool dirty_;
};

template <typename B, typename L, sync_policy S>
class Accessor<B, L, S, remote_access,
	       typename enable_if<!(S&device)>::type>
{
  // Manage data transfer between distributed and local block
  // during object initialization and finalization.
  template <typename LB>
  struct assigner
  {
    assigner(LB &l, B &b) : l_(l), b_(b) 
    {
      l_.admit(false);
      sync_in();
    }
    ~assigner()
    {
      sync_out();
      l_.release(false);
    }
    void sync_in() { if (S&in) parallel::assign_local(l_, b_);}
    void sync_out()
    {
      if (S&out)
	parallel::assign_local_if<is_modifiable_block<B>::value>(b_, l_);
    }
    LB &l_;
    B &b_;
  };

public:
  static access_kind const access = remote_access;
  typedef typename adjust_layout<L, typename get_block_layout<B>::type>::type
    layout_type;
  typedef typename B::value_type value_type;
  typedef typename view_of<B>::type dist_view_type;

  typedef host_storage<value_type, layout_type::storage_format> storage_type;
  typedef typename storage_type::ptr_type non_const_ptr_type;
  typedef typename storage_type::const_ptr_type const_ptr_type;
  typedef typename conditional<is_modifiable_block<B>::value,
			       non_const_ptr_type,
			       const_ptr_type>::type ptr_type;

  typedef Strided<B::dim, value_type, layout_type> local_block_type;
  typedef Accessor<local_block_type, layout_type, S> data_access_type;

  static int const ct_cost = cost<remote_access>::value;

  static int cost(B const &) { return ct_cost;}
  static length_type required_buffer_size(B const &b)
  {
    Applied_layout<layout_type> layout(extent<B::dim>(b));
    return layout.total_size();
  }

  Accessor(B &b, non_const_ptr_type buffer = non_const_ptr_type())
    : block_(b),
      storage_(b.size(), buffer),
      lblock_(block_domain<B::dim>(b), storage_.ptr()),
      assigner_(lblock_, block_),
      local_access_(lblock_)
  {
  }
  ~Accessor() {}

  void sync_in()
  {
    assigner_.sync_in();
    local_access_.sync_in();
  }
  void sync_out()
  {
    local_access_.sync_out();
    assigner_.sync_out();
  }

  int cost() const { return ct_cost;}

  ptr_type ptr() { return local_access_.ptr();}
  const_ptr_type ptr() const { return local_access_.ptr();}
  stride_type stride(dimension_type d) const { return local_access_.stride(d);}
  length_type size(dimension_type d) const { return local_access_.size(d);}
  length_type size() const { return local_access_.size();}
  length_type storage_size() const { return local_access_.storage_size();}

private:
  B &block_;
  storage_type storage_;
  mutable local_block_type lblock_;
  assigner<local_block_type> assigner_;
  data_access_type local_access_;
};

/// Accessor that postpones the access policy until runtime.
/// Depending on the block's stride(s) access will be direct
/// or copy.
template <typename B, typename L, sync_policy S>
class Accessor<B, L, S, maybe_direct_access,
	       typename enable_if<!(S&device)>::type>
{
  typedef typename get_block_layout<B>::type block_layout_type;
public:
  static access_kind const access = maybe_direct_access;
  static dimension_type const dim = L::dim;

  typedef typename B::value_type value_type;
  // The layout type defined here only applies if the block
  // needs to be copied. It may not match the block's layout.
  typedef typename Accessor<B, L, S, copy_access>::layout_type layout_type;
  typedef typename layout_type::order_type order_type;
  static pack_type const packing = layout_type::packing;
  static storage_format_type const storage_format = layout_type::storage_format;
  typedef host_storage<value_type, layout_type::storage_format> storage_type;
  typedef typename storage_type::ptr_type non_const_ptr_type;
  typedef typename storage_type::const_ptr_type const_ptr_type;
  typedef typename conditional<is_modifiable_block<B>::value,
			       non_const_ptr_type,
			       const_ptr_type>::type ptr_type;

  static int   const ct_cost = cost<copy_access>::value;

  static int cost(B const &block) { return is_compatible<L>(block) ? 0 : 2;}
  static std::size_t required_buffer_size(B const &b)
  {
    if (is_compatible<L>(b)) return 0;
    else return Accessor<B, L, S, copy_access>::required_buffer_size(b);
  }

  Accessor(B &b, non_const_ptr_type buffer = non_const_ptr_type())
    : block_(b),
      use_direct_(is_compatible<L>(block_)),
      layout_(extent<dim>(block_)),
      storage_(use_direct_ ? 0 : layout_.total_size(), buffer)
  { sync_in();}

  ~Accessor() { sync_out();}

  void sync_in() 
  {
    if (!use_direct_ && S&in)
      block_copy(block_, storage_.ptr(), layout_);
  }
  void sync_out() 
  {
    if (!use_direct_ && S&out)
      block_copy(storage_.ptr(), layout_, block_);
  }

  B &block() { return block_;}
  B const &block() const { return block_;}
  Applied_layout<layout_type> const &layout() { return layout_;}

  int cost() const { return use_direct_ ? 0 : 2;}

  ptr_type ptr()
  { return use_direct_ ? block_.ptr() : storage_.ptr();}
  const_ptr_type ptr() const
  { return use_direct_ ? block_.ptr() : storage_.ptr();}
  stride_type stride(dimension_type d) const
  { return use_direct_ ? block_.stride(dim, d) : layout_.stride(d);}
  length_type size(dimension_type d) const
  { return use_direct_ ? block_.size(dim, d) : block_.size(B::dim, d);}
  length_type	size  () const { return block_.size();}
  length_type storage_size() const { return layout_.total_size();}

private:
  B &block_;
  bool use_direct_;
  Applied_layout<layout_type> layout_;
  storage_type storage_;
};

template <typename B, dimension_type D, sync_policy S>
class Accessor<B, Rt_layout<D>, S, maybe_direct_access,
	       typename enable_if<!(S&device)>::type>
{
public:
  static access_kind const access = maybe_direct_access;
  static dimension_type const dim = D;

  typedef typename B::value_type value_type;
  static storage_format_type const storage_format =
    is_complex<value_type>::value ? any_storage_format : array;
  typedef host_storage<value_type, storage_format> storage_type;
  typedef typename storage_type::ptr_type non_const_ptr_type;
  typedef typename storage_type::const_ptr_type const_ptr_type;
  typedef typename conditional<is_modifiable_block<B>::value,
			       non_const_ptr_type,
			       const_ptr_type>::type ptr_type;

  static int const ct_cost = cost<copy_access>::value;

  static int cost(B const &block, Rt_layout<D> const &rtl)
  { return is_compatible(block, rtl) ? 0 : ct_cost;}

  Accessor(B &b, Rt_layout<D> const& rtl, bool force_copy)
    : block_(b),
      use_direct_(!force_copy && is_compatible(b, rtl)),
      layout_(use_direct_ ?
	      Applied_layout<Rt_layout<D> >(empty_layout) :
	      Applied_layout<Rt_layout<D> >(rtl, extent<dim>(b), sizeof(value_type))),
      storage_(use_direct_ ? 0 : layout_.total_size(), rtl.storage_format)
  {
    sync_in();
  }

  Accessor(B &b, Rt_layout<D> const& rtl, bool force_copy,
	   non_const_ptr_type buffer, length_type buffer_size)
    : block_(b),
      use_direct_(!force_copy && is_compatible(b, rtl)),
      layout_(use_direct_ ?
	      Applied_layout<Rt_layout<D> >(empty_layout) :
	      Applied_layout<Rt_layout<D> >(rtl, extent<dim>(b), sizeof(value_type))),
      storage_(use_direct_ ? 0 : layout_.total_size(),
	       buffer ? buffer :
	       // yuck !
	       // For complex data we want to pass the storage format as part of the pointer<> instance
	       // For non-complex data we don't need that.
	       storage_format == array ? non_const_ptr_type() :
	       non_const_ptr_type(rtl.storage_format))
  {
    OVXX_PRECONDITION(use_direct_ || buffer_size == 0 || layout_.total_size() <= buffer_size);
    sync_in();
  }
  ~Accessor() { sync_out();}

  void sync_in()
  {
    if (!use_direct_ && S&in)
      block_copy(block_, storage_.ptr(), layout_);
  }
  void sync_out()
  {
    if (!use_direct_ && S&out)
      block_copy(storage_.ptr(), layout_, block_);
  }

  B &block() { return block_;}
  B const &block() const { return block_;}
  Applied_layout<Rt_layout<D> > const &layout() { return layout_;}

  int cost() const { return use_direct_ ? 0 : 2;}

  ptr_type ptr()
  { return use_direct_ ? ptr_type(block_.ptr()) : storage_.ptr();}
  const_ptr_type ptr() const
  { return use_direct_ ? const_ptr_type(block_.ptr()) : storage_.ptr();}

  stride_type stride(dimension_type d) const
  { return use_direct_ ? block_.stride(dim, d) : layout_.stride(d);}

  length_type size(dimension_type d) const
  { return use_direct_ ? block_.size(dim, d) : layout_.size(d);}
  length_type size() const
  { return use_direct_ ? block_.size() : layout_.total_size();}

private:
  B &block_;
  bool use_direct_;
  Applied_layout<Rt_layout<D> > layout_;
  storage_type storage_;
};

template <typename B, dimension_type D, sync_policy S>
class Accessor<B, Rt_layout<D>, S, copy_access,
	       typename enable_if<!(S&device)>::type>
{
public:
  static access_kind const access = copy_access;
  static dimension_type const dim = D;

  typedef typename B::value_type value_type;
  static storage_format_type const storage_format =
    is_complex<value_type>::value ? any_storage_format : array;
  typedef host_storage<value_type, storage_format> storage_type;
  typedef typename storage_type::ptr_type non_const_ptr_type;
  typedef typename storage_type::const_ptr_type const_ptr_type;
  typedef typename conditional<is_modifiable_block<B>::value,
			       non_const_ptr_type,
			       const_ptr_type>::type ptr_type;

  static int const ct_cost = cost<copy_access>::value;

  static int cost(B const &block, Rt_layout<D> const& rtl)
  { return is_compatible(block, rtl) ? 0 : ct_cost;}

  Accessor(B &b, Rt_layout<D> const& rtl, bool /*force_copy*/)
    : block_(b),
      layout_(Applied_layout<Rt_layout<D> >(rtl, extent<dim>(b), sizeof(value_type))),
      storage_(layout_.total_size(), rtl.storage_format)
  {
    sync_in();
  }

  Accessor(B &b, Rt_layout<D> const &rtl, bool /*force_copy*/,
	   non_const_ptr_type buffer, length_type buffer_size)
    : block_(b),
      layout_(Applied_layout<Rt_layout<D> >(rtl, extent<dim>(b), sizeof(value_type))),
      storage_(layout_.total_size(),
	       buffer ? buffer :
	       // yuck !
	       // For complex data we want to pass the storage format as part of the pointer<> instance
	       // For non-complex data we don't need that.
	       storage_format == array ? non_const_ptr_type() :
	       non_const_ptr_type(rtl.storage_format))
  {
    OVXX_PRECONDITION(buffer_size == 0 || layout_.total_size() <= buffer_size);
    sync_in();
  }

  ~Accessor() { sync_out();}

  void sync_in() { if (S&in) block_copy(block_, storage_.ptr(), layout_);}
  void sync_out() { if (S&out) block_copy(storage_.ptr(), layout_, block_);}

  Applied_layout<Rt_layout<D> > const &layout() { return layout_;}

  int cost() const { return ct_cost;}

  ptr_type ptr() { return storage_.ptr();}
  const_ptr_type ptr() const { return storage_.ptr();}
  stride_type stride(dimension_type d) const { return layout_.stride(d);}
  length_type size(dimension_type d) const { return block_.size(B::dim, d);}

private:
  B &block_;
  Applied_layout<Rt_layout<D> > layout_;
  storage_type storage_;
};

} // namespace ovxx::dda
} // namespace ovxx

#endif

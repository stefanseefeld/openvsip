//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_dda_hpp_
#define vsip_dda_hpp_

#include <ovxx/layout.hpp>
#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/dda/accessor.hpp>

namespace vsip
{
namespace dda
{

/// Determine desired block layout.
///
/// For a block with direct access, the desired layout is the same
/// as the block's layout (get_block_layout).
///
/// For a block with copy access, the desired layout adjusts the
/// pack type to be dense, so that the block can be copied into
/// contiguous memory.
template <typename Block>
struct dda_block_layout
{
private:
  typedef typename ovxx::remove_const<Block>::type block_type;
  typedef typename get_block_layout<block_type>::type block_layout_type;

public:
  static dimension_type const dim = block_layout_type::dim;

  typedef typename block_layout_type::order_type order_type;
  static pack_type const packing = 
    supports_dda<block_type>::value
    ? block_layout_type::packing
    : (ovxx::is_packing_aligned<block_layout_type::packing>::value
       ? block_layout_type::packing
       : dense);
  static storage_format_type const storage_format = block_layout_type::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> layout_type;
};

/// Return the required buffer size for DDA of the given block,
/// using a specific layout.
/// Template parameters:
///
///   :L: the requested layout.
///   :B: the block type.
template <typename L, typename B>
length_type required_buffer_size(B const &b, bool forced_copy = false)
{
  if (forced_copy)
    return ovxx::dda::Accessor<B, L, copy>::required_buffer_size(b);
  else
    return ovxx::dda::Accessor<B, L, in>::required_buffer_size(b);
}

/// Return the required buffer size for DDA of the given block.
template <typename B>
length_type required_buffer_size(B const &b, bool forced_copy = false)
{
  return required_buffer_size<typename dda_block_layout<B>::type>(b, forced_copy);
}

/// Direct (block) data accessor.
///
/// Template parameters
///
///   :Block:  the block type.
///   :Sync:   a sync-policy
///   :Layout: the desired layout policy for the data access.
///
/// The general case covers modifiable blocks
template <typename Block,
	  sync_policy Sync,
	  typename Layout = typename dda_block_layout<Block>::layout_type,
	  bool ReadOnly = !(Sync&out),
          bool NonConstAccess = !ReadOnly || (Sync&copy)>
class Data;

// The general case: read-write access
template <typename B, sync_policy S, typename L, bool A>
class Data<B, S, L, false, A> :
  ovxx::detail::noncopyable,
  ovxx::ct_assert<ovxx::is_modifiable_block<B>::value>
{
  typedef ovxx::dda::Accessor<B, L, S> backend_type;

public:
  typedef typename B::value_type value_type;
  typedef typename backend_type::layout_type layout_type;
  typedef typename backend_type::ptr_type ptr_type;
  typedef typename backend_type::const_ptr_type const_ptr_type;

  static int   const ct_cost = backend_type::ct_cost;

  Data(B &block, ptr_type buffer = ptr_type())
    : backend_(block, buffer)
  {}

  ~Data() {}

  void sync_in() { backend_.sync_in();}
  void sync_out() { backend_.sync_out();}

  ptr_type ptr() { return backend_.ptr();}
  const_ptr_type ptr() const { return backend_.ptr();}
  stride_type stride(dimension_type d) const { return backend_.stride(d);}
  length_type size(dimension_type d) const { return backend_.size(d);}
  length_type size() const { return backend_.size();}
  length_type storage_size() const { return backend_.storage_size();}
  int cost() const { return backend_.cost();}

private:
  backend_type backend_;
};

/// Specialization for read-only synchronization
template <typename B, sync_policy S, typename L>
class Data<B, S, L, true, false> : ovxx::detail::noncopyable
{
  typedef typename ovxx::remove_const<B>::type non_const_block_type;
  typedef ovxx::dda::Accessor<B const, L, S> backend_type;

public:
  typedef typename B::value_type value_type;
  typedef typename backend_type::layout_type layout_type;
  typedef typename backend_type::non_const_ptr_type non_const_ptr_type;
  typedef typename backend_type::const_ptr_type ptr_type;
  typedef typename backend_type::const_ptr_type const_ptr_type;

  static int   const ct_cost = backend_type::ct_cost;

  Data(B const &block, non_const_ptr_type buffer = non_const_ptr_type())
    : backend_(const_cast<non_const_block_type &>(block), buffer)
  {}

  ~Data() {}

  void sync_in() { backend_.sync_in();}
  void sync_out() { backend_.sync_out();}

  ptr_type ptr() const { return backend_.ptr();}
  stride_type stride(dimension_type d) const { return backend_.stride(d);}
  length_type size(dimension_type d) const { return backend_.size(d);}
  length_type size() const { return backend_.size();}
  length_type storage_size() const { return backend_.storage_size();}
  int cost() const { return backend_.cost();}

private:
  backend_type backend_;
};

/// Specialization for read-only access with forced copy.
/// This form of DDA provides a non-const pointer type.
template <typename B, sync_policy S, typename L>
class Data<B, S, L, true, true> : ovxx::detail::noncopyable
{
  typedef typename ovxx::remove_const<B>::type non_const_block_type;
  typedef ovxx::dda::Accessor<B const, L, S> backend_type;

public:
  typedef typename B::value_type value_type;
  typedef typename backend_type::layout_type layout_type;
  typedef typename backend_type::non_const_ptr_type ptr_type;
  typedef typename backend_type::const_ptr_type const_ptr_type;

  static int   const ct_cost = backend_type::ct_cost;

  Data(B const &block, ptr_type buffer = ptr_type())
    : backend_(const_cast<non_const_block_type &>(block), buffer)
    {}

  ~Data() {}

  void sync_in() { backend_.sync_in();}
  void sync_out() { backend_.sync_out();}

  ptr_type ptr() const { return ovxx::const_cast_<ptr_type>(backend_.ptr());}
  stride_type stride(dimension_type d) const { return backend_.stride(d);}
  length_type size(dimension_type d) const { return backend_.size(d);}
  length_type size() const { return backend_.size();}
  length_type storage_size() const { return backend_.storage_size();}
  int cost() const { return backend_.cost();}

private:
  backend_type backend_;
};

/// Return the cost of accessing a block with a given layout.
template <typename L, typename Block>
inline int cost(Block const &block, L const & = L())
{
  return ovxx::dda::Accessor<Block, L, in>::cost(block);
}

/// Return the cost of accessing a block
template <typename Block>
inline int cost(Block const &block)
{
  typedef typename get_block_layout<Block>::type layout_type;
  return cost<layout_type>(block);
}

} // namespace vsip::dda
} // namespace vsip

#endif

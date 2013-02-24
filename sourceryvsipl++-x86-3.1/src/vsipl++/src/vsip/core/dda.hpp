/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */
#ifndef vsip_core_dda_hpp_
#define vsip_core_dda_hpp_

#include <vsip/core/static_assert.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/storage.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/block_copy.hpp>
#include <vsip/core/us_block.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip/core/assign_local.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/dda.hpp>

namespace vsip
{
namespace dda
{
namespace impl
{
using namespace vsip::impl;

/// @group Data Access Tags {

/// Direct_access_tag   -- use direct access to block data
///                        (data, stride member functions).
struct Direct_access_tag {};

/// Reorder_access_tag  -- use direct access to block data, but reorder data
///                        to match requested dimension-order.
struct Reorder_access_tag {};

/// Copy_access_tag     -- copy block data (either using direct access if
///                        available, or just get/put).
struct Copy_access_tag {};

/// Flexible_access_tag -- determine whether to use direct or copy access
///                        at runtime.
struct Flexible_access_tag {};

/// }

#if VSIP_IMPL_REF_IMPL
template <typename Block, typename L>
struct Choose_access
{
  typedef typename vsip::impl::remove_const<Block>::type block_type;
  typedef typename get_block_layout<block_type>::type block_layout_type;

  typedef typename 
    conditional<supports_dda<block_type>::value &&
		is_same<block_layout_type, L>::value,
		Direct_access_tag, Copy_access_tag>::type
    type;
};
#endif

struct direct; // Use dda::Data directly on block.
struct local;  // Use dda::Data on get_local_block of block
struct remap;  // Use dda::Data on reorganized block.

template <typename Block, typename L>
struct Choose_impl_tag
{
  static dimension_type const dim = L::dim;
  typedef typename vsip::impl::remove_const<Block>::type block_type;
  typedef typename block_type::value_type value_type;
  typedef typename block_type::map_type map_type;
  typedef typename get_block_layout<block_type>::type block_layout_type;

  static bool const local_equiv =
    is_layout_compatible<value_type, L, block_layout_type>::value &&
    is_same<Replicated_map<dim>, map_type>::value;

  static bool const equiv = local_equiv &&
    adjust_type<Local_map, map_type>::equiv;

  static bool const is_local = is_same<Local_map, map_type>::value;

  typedef typename
    conditional<is_local, direct,
		typename conditional<local_equiv, local,
				     remap>::type>::type
  type;
};

/// Low-level data access class.
///
/// Template parameters:
///
///   :Block: is a block that supports the data access interface indicated
///           by `AT`.
///   :LP:    is a layout policy compatible with access tag `AT` and block 
///           `Block`.
///   :AT: is a valid data access tag,
///
/// (Each specializtion may provide additional requirements).
///
/// Member Functions:
///    ...
///
/// Notes:
///   Accessor does not hold a block reference/pointer, it
///   is provided to each member function by the caller.  This allows
///   the caller to make policy decisions, such as reference counting.
template <typename Block,
	  typename LP,
	  typename AT,
	  typename Impl = typename Choose_impl_tag<Block, LP>::type>
class Accessor;

/// Specialization for low-level direct data access.
///
/// Template parameters:
///   BLOCK to be a block that supports direct access via member
///     functions ptr() and stride().
///   LP is a layout policy describing the desired layout.  It is should
///     match the inherent layout of the block.  Specifying a layout
///     not directly supported by the block is an error and results in
///     undefined behavior.
template <typename Block,
	  typename LP>
class Accessor<Block, LP, Direct_access_tag, direct>
{
  // Compile time typedefs.
public:
  static dimension_type const dim = LP::dim;

  typedef typename Block::value_type value_type;
  typedef typename LP::order_type    order_type;
  static pack_type const packing = LP::packing;
  static storage_format_type const storage_format = LP::storage_format;

  typedef Storage<storage_format, value_type> storage_type;
  typedef typename storage_type::type non_const_ptr_type;
  typedef typename storage_type::const_type const_ptr_type;
  typedef typename 
  vsip::impl::conditional<vsip::impl::is_modifiable_block<Block>::value,
			  non_const_ptr_type,
			  const_ptr_type>::type ptr_type;

  static int   const CT_Cost         = 0;
  static bool  const CT_Mem_not_req  = true;
  static bool  const CT_Xfer_not_req = true;

  static int    cost         (Block const& /*block*/, LP const& /*layout*/)
    { return CT_Cost; }
  static size_t mem_required (Block const& /*block*/, LP const& /*layout*/)
    { return 0; }
  static size_t xfer_required(Block const& /*block*/, LP const& /*layout*/)
    { return !CT_Xfer_not_req; }

  // Constructor and destructor.
public:
  Accessor(Block&, non_const_ptr_type = non_const_ptr_type()) {}
  ~Accessor() {}

  void begin(Block*, bool) {}
  void end(Block*, bool) {}

  int cost() const { return CT_Cost; }

  // Direct data acessors.
public:
  ptr_type ptr(Block * blk) const { return blk->ptr();}
  stride_type	stride(Block* blk, dimension_type d) const { return blk->stride(dim, d);}
  length_type	size  (Block* blk, dimension_type d) const { return blk->size(dim, d);}
  length_type	size  (Block* blk) const { return blk->size();}
};

/// Specialization for distributed blocks with matching layout.
/// Use get_local_block().
template <typename Block,
	  typename LP>
class Accessor<Block, LP, Direct_access_tag, local>
{
  typedef typename remove_const<Block>::type non_const_block_type;
  typedef typename add_const<Block>::type const_block_type;
  typedef typename conditional<is_const<Block>::value,
    typename Distributed_local_block<non_const_block_type>::type const,
    typename Distributed_local_block<non_const_block_type>::type>::type
  local_block_type;

public:
  static dimension_type const dim = LP::dim;

  typedef typename Block::value_type value_type;
  typedef typename LP::order_type    order_type;
  static pack_type const packing = LP::packing;
  static storage_format_type const storage_format = LP::storage_format;

  typedef Storage<storage_format, value_type> storage_type;
  typedef typename storage_type::type non_const_ptr_type;
  typedef typename storage_type::const_type const_ptr_type;
  typedef typename 
  vsip::impl::conditional<vsip::impl::is_modifiable_block<Block>::value,
			  non_const_ptr_type,
			  const_ptr_type>::type ptr_type;

  static int   const CT_Cost         = 0;
  static bool  const CT_Mem_not_req  = true;
  static bool  const CT_Xfer_not_req = true;

  static int    cost(Block const&, LP const&) { return CT_Cost;}
  static size_t mem_required (Block const&, LP const&) { return 0;}
  static size_t xfer_required(Block const&, LP const&) { return !CT_Xfer_not_req;}

  Accessor(Block &b, non_const_ptr_type = non_const_ptr_type())
    : block_(get_local_block(const_cast<non_const_block_type &>(b))) {}
  ~Accessor() {}

  void begin(Block*, bool) {}
  void end(Block*, bool) {}

  int cost() const { return CT_Cost;}

  ptr_type ptr(Block *) const { return block_.ptr();}
  stride_type  stride(Block *, dimension_type d) const { return block_.stride(dim, d);}
  length_type  size(Block *, dimension_type d) const { return block_.size(dim, d);}
  length_type  size(Block *) const { return block_.size();}

private:
  local_block_type &block_;
};



/// Specialization for copied direct data access.
///
/// Template parameters:
///   :Block: to be a block.
///   :LP:    is a layout policy describing the desired layout.
///           The desired layout can be different from the block's layout.
///
/// Notes:
///   When the desired layout packing format is either packing::unit_stride or
///   packing::unknown, the packing format used will be packing::dense.
template <typename Block,
	  typename LP>
class Accessor<Block, LP, Copy_access_tag, direct>
{
  // Compile time typedefs.
public:
  static dimension_type const dim = LP::dim;

  typedef typename Block::value_type value_type;
  typedef typename LP::order_type    order_type;
  static pack_type const packing =
    LP::packing == unit_stride || LP::packing == any_packing
    ? dense : LP::packing;
  static storage_format_type const storage_format = LP::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> actual_layout_type;

  typedef Allocated_storage<storage_format, value_type> storage_type;
  typedef typename storage_type::type non_const_ptr_type;
  typedef typename storage_type::const_type const_ptr_type;
  typedef typename 
  vsip::impl::conditional<vsip::impl::is_modifiable_block<Block>::value,
			  non_const_ptr_type,
			  const_ptr_type>::type ptr_type;

  static int   const CT_Cost          = 2;
  static bool  const CT_Mem_not_req   = false;
  static bool  const CT_Xfer_not_req  = false;

  static int    cost(Block const&, LP const&)
    { return CT_Cost; }
  static size_t mem_required (Block const& block, LP const&)
    { return sizeof(typename Block::value_type) * block.size(); }
  static size_t xfer_required(Block const&, LP const&)
    { return !CT_Xfer_not_req; }

  // Constructor and destructor.
public:
  Accessor(Block & blk, non_const_ptr_type buffer = non_const_ptr_type())
    : layout_   (extent<dim>(blk)),
      storage_  (layout_.total_size(), buffer)
  {}

  ~Accessor()
    { storage_.deallocate(layout_.total_size());}

  void begin(Block* blk, bool sync)
  {
    if (sync)
      Block_copy_to_ptr<LP::dim, Block, order_type, packing, storage_format>::
	copy(blk, layout_, storage_.ptr());
  }

  void end(Block* blk, bool sync)
  {
    if (sync)
      Block_copy_from_ptr<LP::dim, Block, order_type, packing, storage_format>::
	copy(blk, layout_, storage_.ptr());
  }

  int cost() const { return CT_Cost; }

  ptr_type ptr(Block*) { return storage_.ptr();}
  const_ptr_type ptr(Block*) const { return storage_.ptr();}
  stride_type stride(Block*, dimension_type d) const { return layout_.stride(d);}
  length_type size(Block* blk, dimension_type d) const { return blk->size(Block::dim, d);}
  length_type size(Block* blk) const { return blk->size();}

private:
  Applied_layout<actual_layout_type> layout_;
  storage_type                       storage_;
};

template <typename B, typename L, typename A>
class Accessor<B, L, A, remap>
{
public:
  typedef typename B::value_type value_type;
  typedef typename view_of<B>::type dist_view_type;

  typedef Allocated_storage<L::storage_format, value_type> storage_type;
  typedef typename storage_type::type non_const_ptr_type;
  typedef typename storage_type::const_type const_ptr_type;
  typedef typename 
  vsip::impl::conditional<vsip::impl::is_modifiable_block<B>::value,
			  non_const_ptr_type,
			  const_ptr_type>::type ptr_type;

  typedef Us_block<B::dim, value_type, L, Local_map> block_type;
  typedef typename view_of<block_type>::type local_view_type;
  typedef Accessor<block_type, L, A> data_access_type;

public:
  static int   const CT_Cost          = 2;
  static bool  const CT_Mem_not_req   = false;
  static bool  const CT_Xfer_not_req  = false;

  static int    cost(B const&, L const&)
  { return CT_Cost;}
  static size_t mem_required (B const & block, L const&)
  { return sizeof(typename B::value_type) * block.size();}
  static size_t xfer_required(B const &, L const &)
  { return !CT_Xfer_not_req;}

  Accessor(B &b, non_const_ptr_type buffer = non_const_ptr_type())
    : storage_(b.size(), buffer),
      block_(block_domain<B::dim>(b), storage_.ptr()),
      ext_(block_)
  {}
  ~Accessor()
  { storage_.deallocate(block_.size());}

  void begin(B *b, bool sync)
  {
    if (sync) assign_local(block_, *b);
    ext_.begin(&block_, sync);
  }
  void end(B *b, bool sync)
  {
    ext_.end(&block_, sync);
    if (sync) assign_local_if<is_modifiable_block<B>::value>(*b, block_);
  }

  int cost() const { return CT_Cost;}

  ptr_type ptr(B*) { return ext_.ptr(&block_);}
  const_ptr_type ptr(B*) const { return ext_.ptr(&block_);}
  stride_type stride(B*, dimension_type d) const { return ext_.stride(&block_, d);}
  length_type size(B *b, dimension_type d) const { return ext_.size(&block_, d);}
  length_type size(B *b) const { return ext_.size(&block_);}

private:
  storage_type storage_;
  mutable block_type block_;
  data_access_type ext_;
};

template <typename AT> struct Cost { static int const value = 10; };
template <> struct Cost<Direct_access_tag>   { static int const value = 0; };
template <> struct Cost<Copy_access_tag>     { static int const value = 2; };

} // namespace vsip::dda::impl
} // namespace vsip::dda
} // namespace vsip

#endif

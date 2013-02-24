/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/extdata.hpp
    @author  Jules Bergmann
    @date    2005-02-11
    @brief   VSIPL++ Library: Core Direct Data Access.

*/

#ifndef VSIP_CORE_EXTDATA_HPP
#define VSIP_CORE_EXTDATA_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/static_assert.hpp>
#include <vsip/core/extdata_common.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/storage.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/block_copy.hpp>
#if !VSIP_IMPL_REF_IMPL
#  include <vsip/opt/extdata.hpp>
#endif



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

/// Reference Counting Policies.
///
/// A reference counting policy describes which behavior should be
/// taken when an DDI class creates and destroys references to a block. 
///
/// A policy should implement two static functions:
///  - inc() - action to perform when reference is created 
///  - dec() - action to perform when reference is destroyed 
///
///
/// The following policies are available:
///
///  - No_count_policy: do not increment/decrement block reference count.
///
///    When a DDI class is used in the same scope as another reference
///    to the block being accessed, it is not necessary to increment
///    the reference count.
///
///  - Ref_count_policy: increment/decrement block reference count.
///
///    When a DDI class is used in a situation where the block it
///    references does not have a guarenteed reference.


/// No reference count policy, indicates DDI object will not increment
/// and decrement reference count.
struct No_count_policy
{
  template <typename Block>
  static void inc(Block const* /* block */) {}

  template <typename Block>
  static void dec(Block const* /* block */) {}
};



/// Reference count policy, indicates DDI object will increment and
/// decrement reference count.
struct Ref_count_policy
{
  template <typename Block>
  static void inc(Block const* block) { block->increment_count(); }

  template <typename Block>
  static void dec(Block const* block) { block->decrement_count(); }
};



/// Namespace for low-level data access interfaces.  These interfaces
/// provide low-level data access to data stored within blocks
/// (directly or indirectly).
///
/// These interfaces are not intended to be used in application code,
/// or in the library implementation outside of the Ext_data class.
/// Not all low-level interfaces are valid for all blocks, and over time
/// details of the low-level interface may change.  To provide a
/// consistent data interface to all blocks, the Ext_data class should
/// be used instead.
namespace data_access 
{

/// Specialization for low-level direct data access.
///
/// Template parameters:
///   BLOCK to be a block that supports direct access via member
///     functions impl_data() and impl_stride().  Access to these
///     members can be protected by making Low_level_data_access a friend
///     class to the block.
///   LP is a layout policy describing the desired layout.  It is should
///     match the inherent layout of the block.  Specifying a layout
///     not directly supported by the block is an error and results in
///     undefined behavior.
template <typename Block,
	  typename LP>
class Low_level_data_access<Direct_access_tag, Block, LP>
{
  // Compile time typedefs.
public:
  static dimension_type const dim = LP::dim;

  typedef typename Block::value_type value_type;
  typedef typename LP::order_type    order_type;
  typedef typename LP::pack_type     pack_type;
  typedef typename LP::complex_type  complex_type;

  typedef Storage<complex_type, value_type> storage_type;
  typedef typename storage_type::type raw_ptr_type;
  typedef typename storage_type::const_type const_raw_ptr_type;

  // Compile- and run-time properties.
public:
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
  Low_level_data_access(Block&,
			raw_ptr_type     = NULL)
  {}

  ~Low_level_data_access() {}

  void begin(Block*, bool) {}
  void end(Block*, bool) {}

  int cost() const { return CT_Cost; }

  // Direct data acessors.
public:
  raw_ptr_type data(Block * blk) const
  { return blk->impl_data();}
  stride_type	stride(Block* blk, dimension_type d) const
    { return blk->impl_stride(dim, d); }
  length_type	size  (Block* blk, dimension_type d) const
    { return blk->size(dim, d); }
  length_type	size  (Block* blk) const
    { return blk->size(); }
};



/// Specialization for copied direct data access.
///
/// Template parameters:
///   :Block: to be a block.
///   :LP:    is a layout policy describing the desired layout.
///           The desired layout can be different from the block's layout.
///
/// Notes:
///   When the desired layout packing format is either Stride_unit or
///   Stride_unknown, the packing format used will be Stride_unit_dense.
template <typename Block,
	  typename LP>
class Low_level_data_access<Copy_access_tag, Block, LP>
{
  // Compile time typedefs.
public:
  static dimension_type const dim = LP::dim;

  typedef typename Block::value_type value_type;
  typedef typename LP::order_type    order_type;
  typedef typename
          ITE_Type<Type_equal<typename LP::pack_type, Stride_unit>::value ||
	           Type_equal<typename LP::pack_type, Stride_unknown>::value,
                   As_type<Stride_unit_dense>,
		   As_type<typename LP::pack_type> >::type pack_type;
  typedef typename LP::complex_type  complex_type;

  typedef Layout<dim, order_type, pack_type, complex_type> actual_layout_type;

  typedef Allocated_storage<complex_type, value_type> storage_type;
  typedef typename storage_type::type                 raw_ptr_type;
  typedef typename storage_type::const_type           const_raw_ptr_type;

  // Compile- and run-time properties.
public:
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
  Low_level_data_access(Block&         blk,
			raw_ptr_type   buffer = NULL)
    : layout_   (extent<dim>(blk)),
      storage_  (layout_.total_size(), buffer)
  {}

  ~Low_level_data_access()
    { storage_.deallocate(layout_.total_size()); }

  void begin(Block* blk, bool sync)
  {
    if (sync)
      Block_copy_to_ptr<LP::dim, Block, order_type, pack_type, complex_type>::
	copy(blk, layout_, storage_.data());
  }

  void end(Block* blk, bool sync)
  {
    if (sync)
      Block_copy_from_ptr<LP::dim, Block, order_type, pack_type, complex_type>::
	copy(blk, layout_, storage_.data());
  }

  int cost() const { return CT_Cost; }

  // Direct data acessors.
public:
  raw_ptr_type	data(Block*)
    { return storage_.data(); }
  const_raw_ptr_type	data(Block*) const
    { return storage_.data(); }
  stride_type	stride(Block*, dimension_type d) const
    { return layout_.stride(d); }
  length_type	size  (Block* blk, dimension_type d) const
    { return blk->size(Block::dim, d); }
  length_type	size  (Block* blk) const
    { return blk->size(); }

  // Member data.
private:
  Applied_layout<actual_layout_type> layout_;
  storage_type                       storage_;
};



template <> struct Cost<Direct_access_tag>   { static int const value = 0; };
template <> struct Cost<Copy_access_tag>     { static int const value = 2; };

} // namespace vsip::impl::data_access



/// Choose access type for a given block and desired layout.

#if VSIP_IMPL_REF_IMPL
template <typename Block,
	  typename LP>
struct Choose_access
{
  typedef typename Block_layout<Block>::layout_type BLP;
  typedef typename Block_layout<Block>::access_type access_type;

  typedef typename 
    ITE_Type<Type_equal<BLP, LP>::value,
	   As_type<access_type>, As_type<Copy_access_tag> >::type
    type;
};
#endif



/// Determine desired block layout.
///
/// For a block with direct access, the desired layout is the same
/// as the block's layout (Block_layout).
///
/// For a block with copy access, the desired layout adjusts the
/// pack type to be dense, so that the block can be copied into
/// contiguous memory.
template <typename BlockT>
struct Desired_block_layout
{
private:
  typedef Block_layout<BlockT> raw_type;

public:
  static dimension_type const dim = raw_type::dim;

  typedef typename raw_type::access_type   access_type;
  typedef typename raw_type::order_type    order_type;
  typedef typename
	  ITE_Type<Type_equal<access_type, Direct_access_tag>::value,
		   As_type<typename raw_type::pack_type>,
	  ITE_Type<Type_equal<access_type, Copy_access_tag>::value &&
                   Is_stride_unit_align<typename raw_type::pack_type>::value,
		      As_type<typename raw_type::pack_type>,
		      As_type<Stride_unit_dense>
	          > >::type                  pack_type;
  typedef typename raw_type::complex_type  complex_type;

  typedef Layout<dim, order_type, pack_type, complex_type> layout_type;
};



/// High-level data access class.  Provides data access to data
/// stored in blocks, using an appropriate low-level data interface.
///
/// Template parameters:
///
///   :BlockT: is a block type.
///   :LP:     is the desired layout policy for the data access.
///   :RP:     is a reference counting policy.
///   :AT:     is a data access tag that selects the low-level interface
///            used to access the data.  By default, Choose_access is used to
///            select the appropriate access tag for a given block type
///            `BlockT` and layout `LP`.
///
/// Notes:
///   Selecting a specific low-level interface is discouraged.
///   Selecting one that is not compatible with BLOCK will result in
///   undefined behavior.
template <typename BlockT,
	  typename LP  = typename Desired_block_layout<BlockT>::layout_type,
	  typename RP  = No_count_policy,
	  typename AT  = typename Choose_access<BlockT, LP>::type>
class Ext_data : Assert_proper_block<BlockT>
{
  // Compile time typedefs.
public:
  typedef typename Non_const_of<BlockT>::type non_const_block_type;

  typedef data_access::Low_level_data_access<AT, BlockT, LP> ext_type;
  typedef typename BlockT::value_type value_type;

  typedef Storage<typename LP::complex_type, typename BlockT::value_type>
		storage_type;

  typedef typename storage_type::alloc_type element_type;
  // The following line should actually be conditionalized on the block's
  // mutability. However, that appears to cause ICEs with various versions
  // of G++.
  typedef typename storage_type::type raw_ptr_type;
  typedef typename storage_type::const_type const_raw_ptr_type;


  // Compile- and run-time properties.
public:
  static int   const CT_Cost          = ext_type::CT_Cost;
  static bool  const CT_Mem_not_req   = ext_type::CT_Mem_not_req;
  static bool  const CT_Xfer_not_req  = ext_type::CT_Xfer_not_req;


  // Constructor and destructor.
public:
  Ext_data(non_const_block_type& block,
	   sync_action_type      sync   = SYNC_INOUT,
	   raw_ptr_type          buffer = storage_type::null())
    : blk_ (&block),
      ext_ (block, buffer),
      sync_(sync)
    { ext_.begin(blk_.get(), sync_ & SYNC_IN); }

  Ext_data(BlockT const&      block,
	   sync_action_type   sync   = SYNC_IN,
	   raw_ptr_type       buffer = storage_type::null())
    : blk_ (&const_cast<BlockT&>(block)),
      ext_ (const_cast<BlockT&>(block), buffer),
      sync_(sync)
  {
    assert(sync != SYNC_OUT && sync != SYNC_INOUT);
    ext_.begin(blk_.get(), sync_ & SYNC_IN);
  }

  ~Ext_data()
    { ext_.end(blk_.get(), sync_ & SYNC_OUT); }

  // Direct data acessors.
public:
  raw_ptr_type data()
    { return ext_.data  (blk_.get()); }

  const_raw_ptr_type data() const
    { return ext_.data  (blk_.get()); }

  stride_type stride(dimension_type d) const
    { return ext_.stride(blk_.get(), d); }

  length_type size(dimension_type d) const
    { return ext_.size  (blk_.get(), d); }

  length_type size() const
    { return ext_.size  (blk_.get()); }

  int cost() const
    { return ext_.cost(); }

  // Member data.
private:
  typename View_block_storage<BlockT>::template With_rp<RP>::type
		   blk_;
  ext_type         ext_;
  sync_action_type sync_;
};


template <typename Block,
	  typename LP  = typename Desired_block_layout<Block>::layout_type,
	  typename RP  = No_count_policy,
	  typename AT  = typename Choose_access<Block, LP>::type>
class Persistent_ext_data : Assert_proper_block<Block>
{
  // Compile time typedefs.
public:
  typedef data_access::Low_level_data_access<AT, Block, LP> ext_type;
  typedef typename Block::value_type value_type;

  typedef Storage<typename LP::complex_type, typename Block::value_type>
		storage_type;

  typedef typename storage_type::alloc_type element_type;
  // The following line should actually be conditionalized on the block's
  // mutability. However, that appears to cause ICEs with various versions
  // of G++.
  typedef typename storage_type::type       raw_ptr_type;
  typedef typename storage_type::const_type const_raw_ptr_type;


  // Compile- and run-time properties.
public:
  static int   const CT_Cost          = ext_type::CT_Cost;
  static bool  const CT_Mem_not_req   = ext_type::CT_Mem_not_req;
  static bool  const CT_Xfer_not_req  = ext_type::CT_Xfer_not_req;


  // Constructor and destructor.
public:
  Persistent_ext_data(Block&             block,
		      sync_action_type   sync   = SYNC_INOUT,
		      raw_ptr_type       buffer = storage_type::null())
    : blk_ (&block),
      ext_ (block, buffer),
      sync_(sync)
    {}

  ~Persistent_ext_data()
    {}

  void begin()
    { ext_.begin(blk_.get(), sync_ & SYNC_IN); }

  void end()
    { ext_.end(blk_.get(), sync_ & SYNC_OUT); }

  // Direct data acessors.
public:
  raw_ptr_type data()
    { return ext_.data  (blk_.get()); }

  const_raw_ptr_type data() const
    { return ext_.data  (blk_.get()); }

  stride_type stride(dimension_type d) const
    { return ext_.stride(blk_.get(), d); }

  length_type size(dimension_type d) const
    { return ext_.size  (blk_.get(), d); }

  int cost() const
    { return ext_.cost(); }

  // Member data.
private:
  typename View_block_storage<Block>::template With_rp<RP>::type
		   blk_;
  ext_type         ext_;
  sync_action_type sync_;
};



template <typename Block,
	  typename LP = typename Desired_block_layout<Block>::layout_type>
struct Ext_data_cost
{
  typedef typename Choose_access<Block, LP>::type access_type;
  static int const value = data_access::Cost<access_type>::value;
};



/***********************************************************************
  Definitions
***********************************************************************/

/// Return the cost of accessing a block with a given layout.
template <typename LP,
	  typename Block>
inline 
int
cost(
  Block const& block,
  LP    const& layout = LP())
{
  typedef typename Choose_access<Block, LP>::type
		access_type;

  return data_access::Low_level_data_access<access_type, Block, LP>
    ::cost(block, layout);
}



/// Return the number of bytes of memory required to access a block
/// with a given layout.
template <typename LP,
	  typename Block>
size_t
mem_required(
  Block const& block,
  LP    const& layout = LP())
{
  typedef typename Choose_access<Block, LP>::type
		access_type;

  return data_access::Low_level_data_access<access_type, Block, LP>
    ::mem_required(block, layout);
}



/// Return whether a transfer is required to access a block with
/// a given layout.
template <typename LP,
	  typename Block>
bool
xfer_required(
  Block const& block,
  LP    const& layout = LP())
{
  typedef typename Choose_access<Block, LP>::type
		access_type;

  return data_access::Low_level_data_access<access_type, Block, LP>
    ::xfer_required(block, layout);
}



/// Determine if an Ext_data object refers to a dense (contiguous,
/// unit-stride) region of memory.
template <typename OrderT,
	  typename ExtT>
bool
is_ext_dense(
  vsip::dimension_type dim,
  ExtT const&          ext)
{
  using vsip::dimension_type;
  using vsip::stride_type;

  dimension_type const dim0 = OrderT::impl_dim0;
  dimension_type const dim1 = OrderT::impl_dim1;
  dimension_type const dim2 = OrderT::impl_dim2;

  assert(dim <= VSIP_MAX_DIMENSION);

  if (dim == 1)
  {
    return (ext.stride(dim0) == 1);
  }
  else if (dim == 2)
  {
    return (ext.stride(dim1) == 1) &&
           (ext.stride(dim0) == static_cast<stride_type>(ext.size(dim1)) ||
	    ext.size(dim0) == 1);
  }
  else /*  if (dim == 2) */
  {
    return (ext.stride(dim2) == 1) &&
           (ext.stride(dim1) == static_cast<stride_type>(ext.size(dim2)) ||
	    (ext.size(dim0) == 1 && ext.size(dim1) == 1)) &&
           (ext.stride(dim0) == static_cast<stride_type>(ext.size(dim1)  *
							 ext.size(dim2)) ||
	    ext.size(dim0) == 1);
  }
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_EXTDATA_HPP

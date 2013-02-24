/* Copyright (c) 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/extdata-dist.hpp
    @author  Jules Bergmann
    @date    2006-01-31
    @brief   VSIPL++ Library: Direct Data Access to Distributed blocks.

*/

#ifndef VSIP_CORE_EXTDATA_DIST_HPP
#define VSIP_CORE_EXTDATA_DIST_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/static_assert.hpp>
#include <vsip/core/block_copy.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/extdata.hpp>
#include <vsip/core/working_view.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/core/us_block.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace edl_details
{

/// Implementation tags for Ext_data local

struct Impl_use_direct; // Use Ext_data directly on block.
struct Impl_use_local;  // Use Ext_data on get_local_block of block
struct Impl_remap;      // Use Ext_data on reorganized block.



/// Choose Ext_data_dist implementation tag for a block.

/// Requires:
///   BLOCKT is a block type to use direct access on,
///   LP     is the requested layout policy.

template <typename BlockT,
	  typename LP>
struct Choose_impl_tag
{
  static dimension_type const dim = LP::dim;
  typedef typename BlockT::value_type                value_type;
  typedef typename BlockT::map_type                  map_type;
  typedef typename Block_layout<BlockT>::layout_type actual_LP;

  static bool const local_equiv =
	Is_layout_compatible<value_type, LP, actual_LP>::value &&
        Type_equal<Global_map<dim>, map_type>::value;

  static bool const equiv = local_equiv &&
	Adjust_type<Local_map, map_type>::equiv;

  static bool const is_local = Type_equal<Local_map, map_type>::value;

  typedef typename
    ITE_Type<is_local,    As_type<Impl_use_direct>,
    ITE_Type<local_equiv, As_type<Impl_use_local>,
		          As_type<Impl_remap> > >
    ::type type;
};

} // namespace vsip::impl::edl_details



/// High-level data access class, reorganizes distributed data to
/// a single local copy.  Provides data access to data stored in
/// blocks, using an appropriate low-level data interface.

/// Requires:
///   BLOCK is a block type.
///   LP is the desired layout policy for the data access.
///   RP is a reference counting policy.
///   AT is a data access tag that selects the low-level interface
///      used to access the data.  By default, Choose_access is used to
///      select the appropriate access tag for a given block type
///      BLOCK and layout LP. [2]
///   IMPLTAG is a tag to choose how block needs to be reorganized,
///      if at all.
///
/// Notes:
/// [1] Selecting a specific low-level interface is discouraged.
///     Selecting one that is not compatible with BLOCK will result in
///     undefined behavior.
///
/// [2] Choose_access is not used for the default template parameter,
///     because the block type may change before access is done, esp
///     when ImplTag = remap.  Instead, "Default_access_type" is used
///     to indicate that Choose_access should be used once the block
///     type is known.

template <typename Block,
	  sync_action_type SP,
	  typename LP      = typename Desired_block_layout<Block>::layout_type,
	  typename RP      = No_count_policy,
	  typename AT      = Default_access_tag,
	  typename ImplTag = typename 
                             edl_details::Choose_impl_tag<Block, LP>::type>
class Ext_data_dist;



/// Helper class for Impl_use_direct variant of Ext_data_dist to determine
/// base class to derive from.

template <typename BlockT,
	  typename LP,
	  typename RP,
	  typename AT>
struct Use_direct_helper
{
  typedef typename
          ITE_Type<Type_equal<AT, Default_access_tag>::value,
                   Choose_access<BlockT, LP>, As_type<AT> >::type
          access_type;
  typedef Ext_data<BlockT, LP, RP, access_type> base_type;
};



/// Ext_data_dist variant to directly use Ext_data for access to block.

template <typename BlockT,
	  sync_action_type SP,
	  typename LP,
	  typename RP,
	  typename AT>
class Ext_data_dist<BlockT, SP, LP, RP, AT, edl_details::Impl_use_direct>
  : public Use_direct_helper<BlockT, LP, RP, AT>::base_type
{
  typedef typename Non_const_of<BlockT>::type non_const_block_type;
  typedef typename Use_direct_helper<BlockT, LP, RP, AT>::base_type base_type;

  typedef typename base_type::storage_type storage_type;
  typedef typename base_type::raw_ptr_type raw_ptr_type;

  // Constructor and destructor.
public:
  Ext_data_dist(non_const_block_type& block,
		raw_ptr_type          buffer = raw_ptr_type())
    : base_type(block, SP, buffer)
  {}

  Ext_data_dist(BlockT const&      block,
		raw_ptr_type       buffer = raw_ptr_type())
    : base_type(block, SP, buffer)
  {}

  ~Ext_data_dist() {}
};



/// Helper class for Impl_use_local variant of Ext_data_dist to determine
/// base class to derive from.

template <typename BlockT,
	  typename LP,
	  typename RP,
	  typename AT>
struct Use_local_helper
{
  typedef typename Distributed_local_block<BlockT>::type local_block_type;
  typedef typename
          ITE_Type<Type_equal<AT, Default_access_tag>::value,
                   Choose_access<local_block_type, LP>, As_type<AT> >::type
          access_type;
  typedef Ext_data<local_block_type, LP, RP, access_type> base_type;
};



/// Ext_data_dist variant to use Ext_data access on a distributed block's
/// local block (as returned by get_local_block).

template <typename         BlockT,
	  sync_action_type SP,
	  typename         LP,
	  typename         RP,
	  typename         AT>
class Ext_data_dist<BlockT, SP, LP, RP, AT, edl_details::Impl_use_local>
  : public Use_local_helper<BlockT, LP, RP, AT>::base_type
{
  typedef typename Non_const_of<BlockT>::type non_const_block_type;
  typedef typename Distributed_local_block<BlockT>::type local_block_type;
  typedef typename Use_local_helper<BlockT, LP, RP, AT>::base_type base_type;

  typedef typename base_type::storage_type storage_type;
  typedef typename base_type::raw_ptr_type raw_ptr_type;

  // Constructor and destructor.
public:
  Ext_data_dist(non_const_block_type& block,
		raw_ptr_type          buffer = raw_ptr_type())
    : base_type(get_local_block(block), SP, buffer)
  {}

  Ext_data_dist(BlockT const&      block,
		raw_ptr_type       buffer = raw_ptr_type())
    : base_type(get_local_block(block), SP, buffer)
  {}

  ~Ext_data_dist() {}
};



/// Ext_data_dist variant to use Ext_data access on a reorganized
/// copy of the original distributed block.

template <typename         BlockT,
	  sync_action_type SP,
	  typename         LP,
	  typename         RP,
	  typename         AT>
class Ext_data_dist<BlockT, SP, LP, RP, AT, edl_details::Impl_remap>
{
  typedef typename Non_const_of<BlockT>::type non_const_block_type;
  static dimension_type const dim = BlockT::dim;
  typedef typename BlockT::value_type value_type;
  typedef Us_block<dim, value_type, LP, Local_map> block_type;
  typedef typename View_of_dim<dim, value_type, block_type>::type view_type;

  typedef typename View_of_dim<dim, value_type, BlockT>::type src_view_type;

  typedef typename
          ITE_Type<Type_equal<AT, Default_access_tag>::value,
                   Choose_access<block_type, LP>, As_type<AT> >::type
          access_type;
  typedef Persistent_ext_data<block_type, LP, RP, access_type> ext_type;

  typedef Allocated_storage<typename LP::complex_type,
			    typename BlockT::value_type> storage_type;
  typedef typename ext_type::raw_ptr_type raw_ptr_type;

  // Constructor and destructor.
public:
  Ext_data_dist(non_const_block_type& block,
		raw_ptr_type          buffer = raw_ptr_type())
    : src_     (block),
      storage_ (block.size(), buffer),
      block_   (block_domain<dim>(block), storage_.data()),
      view_    (block_),
      ext_     (block_, SP)
  {
    assign_local(view_, src_);
    ext_.begin();
  }

  Ext_data_dist(BlockT const&      block,
		raw_ptr_type       buffer = raw_ptr_type())
    : src_     (const_cast<BlockT&>(block)),
      storage_ (block.size(), buffer),
      block_   (block_domain<dim>(block), storage_.data()),
      view_    (block_),
      ext_     (block_, SP)
  {
    assert(SP != SYNC_OUT && SP != SYNC_INOUT);
    assign_local(view_, src_);
    ext_.begin();
  }

  ~Ext_data_dist()
  {
    ext_.end();
    assign_local_if<SP & SYNC_OUT>(src_, view_);
    storage_.deallocate(block_.size());
  }


  // Direct data acessors.
public:
  raw_ptr_type	data  ()                 { return ext_.data  ();  }
  stride_type	stride(dimension_type d) { return ext_.stride(d); }
  length_type	size  (dimension_type d) { return ext_.size  (d); }

  // Copy to temp buffer view_/block_ forces cost = 2.
  int           cost  ()                 { return 2; }

private:
  src_view_type    src_;	// view of source block
  storage_type     storage_;	// buffer
  block_type       block_;	// Us_block referring to storage_ buffer.
  view_type        view_;
  ext_type         ext_;
};



} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_EXTDATA_DIST_HPP

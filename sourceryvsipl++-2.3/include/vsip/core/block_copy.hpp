/* Copyright (c) 2005, 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/block_copy.hpp
    @author  Jules Bergmann
    @date    2005-02-11
    @brief   VSIPL++ Library: Copy block data into or out of regular memory.
*/

#ifndef VSIP_CORE_BLOCK_COPY_HPP
#define VSIP_CORE_BLOCK_COPY_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/storage.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/parallel/map_traits.hpp>
#include <vsip/core/expr/evaluation.hpp>
#include <vsip/core/profile.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

/// Utility class to copy block data into/out-of of regular memory.
///
/// Template parameters:
///   :Dim:
///   :Block:    is a block type,
///   :Order:    is a dimension-ordering,
///   :PackType: is a packing type tag,
///   :CmplxFmt: is a complex storage format tag.
template <dimension_type Dim,
	  typename       Block,
	  typename       Order,
	  typename       PackType,
	  typename       CmplxFmt>
struct Block_copy_to_ptr
{
  typedef Applied_layout<Layout<Dim, Order, PackType, CmplxFmt> > LP;
  typedef Storage<CmplxFmt, typename Block::value_type> storage_type;
  typedef typename storage_type::type ptr_type;


  static void copy (Block* block, LP& layout, ptr_type data)
  {
    using namespace vsip::impl::profile;
    event<memory>("Block_copy_to_ptr::copy",
		  block->size() * sizeof(typename Block::value_type));
    Length<Dim> ext = extent<Dim>(*block);
    expr::evaluate(*block);
    for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
      storage_type::put(data, layout.index(idx), get(*block, idx));
  }
};



template <dimension_type Dim,
	  typename       Block,
	  typename       Order,
	  typename       PackType,
	  typename       CmplxFmt,
	  bool           Modifiable = Is_modifiable_block<Block>::value>
struct Block_copy_from_ptr
{
  typedef Applied_layout<Layout<Dim, Order, PackType, CmplxFmt> > LP;
  typedef Storage<CmplxFmt, typename Block::value_type> storage_type;
  typedef typename storage_type::type ptr_type;

  static void copy(Block* block, LP& layout, ptr_type data)
  {
    using namespace vsip::impl::profile;
    event<memory>("Block_copy_from_ptr::copy",
		  block->size() * sizeof(typename Block::value_type));
    Length<Dim> ext = extent<Dim>(*block);

    for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
    {
      put(*block, idx, storage_type::get(data, layout.index(idx)));
    }
  }
};



template <dimension_type Dim,
	  typename       Block,
	  typename       Order,
	  typename       PackType,
	  typename       CmplxFmt>
struct Block_copy_from_ptr<Dim, Block, Order, PackType, CmplxFmt, false>
{
  typedef Applied_layout<Layout<Dim, Order, PackType, CmplxFmt> > LP;
  typedef Storage<CmplxFmt, typename Block::value_type> storage_type;
  typedef typename storage_type::type ptr_type;

  static void copy(Block*, LP&, ptr_type)
  { assert(0); }
};



/// Implementation class to copy block data to pointer to regular memory,
/// with layout determined at run-time.
///
/// Template parameters:
///
///   :Dim: is the dimension of the run-time layout.
///   :Block: is a block type,
///   :IsComplex: indicates whether the value type is complex
///
/// IsComplex is used to specialize the implementation so that functions
/// can be overloaded for split and interleaved arguments.

template <dimension_type Dim,
	  typename       Block,
	  bool           IsComplex>
struct Rt_block_copy_to_ptr_impl;

/// Specialization for blocks with complex value_types.
template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_to_ptr_impl<Dim, Block, true>
{
  typedef Applied_layout<Rt_layout<Dim> > LP;

  typedef Storage<Cmplx_inter_fmt, typename Block::value_type>
	  inter_storage_type;
  typedef Storage<Cmplx_split_fmt, typename Block::value_type>
	  split_storage_type;

  typedef typename inter_storage_type::type inter_ptr_type;
  typedef typename split_storage_type::type split_ptr_type;

  typedef Rt_pointer<typename Block::value_type> rt_ptr_type;

  static void copy(Block* block, LP const& layout, inter_ptr_type data)
  {
    Length<Dim> ext = extent<Dim>(*block);

    for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
      inter_storage_type::put(data, layout.index(idx), get(*block, idx));
  }

  static void copy(Block* block, LP const& layout, split_ptr_type data)
  {
    Length<Dim> ext = extent<Dim>(*block);

    for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
      split_storage_type::put(data, layout.index(idx), get(*block, idx));
  }

  static void copy(Block* block, LP const& layout, rt_ptr_type data)
  {
    if (complex_format(layout) == cmplx_inter_fmt)
      copy(block, layout, data.as_inter());
    else
      copy(block, layout, data.as_split());
  }
};

/// Specialization for blocks with non-complex value_types.
template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_to_ptr_impl<Dim, Block, false>
{
  typedef Applied_layout<Rt_layout<Dim> > LP;

  typedef Storage<Cmplx_inter_fmt, typename Block::value_type>
	  inter_storage_type;

  typedef typename inter_storage_type::type inter_ptr_type;
  typedef Rt_pointer<typename Block::value_type> rt_ptr_type;


  static void copy(Block* block, LP const& layout, inter_ptr_type data)
  {
    Length<Dim> ext = extent<Dim>(*block);

    for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
      inter_storage_type::put(data, layout.index(idx), get(*block, idx));
  }

  static void copy(Block* block, LP const& layout, rt_ptr_type data)
  {
    copy(block, layout, data.as_inter());
  }

};



/// Implementation class to copy block data from pointer to regular memory,
/// with layout determined at run-time.
///
/// Template parameters:
///
///   :Dim: is the dimension of the run-time layout.
///   :Block: is a block type,
///   :IsComplex: indicates whether the value type is complex
///   :Modifiable: indicates whether the block is modifiable
///
/// IsComplex is used to specialize the implementation so that functions
/// can be overloaded for split and interleaved arguments.
///
/// Modifiable is used to convert attempting to write to a
/// non-modifiable block from a compile-time error to a run-time
/// error.  This is necessary because Ext_data passes sync as
/// a run-time parameter.
template <dimension_type Dim,
	  typename       Block,
	  bool           IsComplex,
	  bool           Modifiable = Is_modifiable_block<Block>::value>
struct Rt_block_copy_from_ptr_impl;

/// Specializations for blocks with complex value_types.
template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_from_ptr_impl<Dim, Block, true, true>
{
  typedef Applied_layout<Rt_layout<Dim> > LP;

  typedef Storage<Cmplx_inter_fmt, typename Block::value_type>
	  inter_storage_type;
  typedef Storage<Cmplx_split_fmt, typename Block::value_type>
	  split_storage_type;

  typedef typename inter_storage_type::type inter_ptr_type;
  typedef typename split_storage_type::type split_ptr_type;

  typedef Rt_pointer<typename Block::value_type> rt_ptr_type;

  static void copy(Block* block, LP const& layout, inter_ptr_type data)
  {
    Length<Dim> ext = extent<Dim>(*block);

    for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
      put(*block, idx, inter_storage_type::get(data, layout.index(idx)));
  }

  static void copy(Block* block, LP const& layout, split_ptr_type data)
  {
    Length<Dim> ext = extent<Dim>(*block);

    for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
      put(*block, idx, split_storage_type::get(data, layout.index(idx)));
  }

  static void copy(Block* block, LP const& layout, rt_ptr_type data)
  {
    if (complex_format(layout) == cmplx_inter_fmt)
      copy(block, layout, data.as_inter());
    else
      copy(block, layout, data.as_split());
  }
};


/// Specialization for const blocks (with complex value type).
///
/// Non-modifiable blocks cannot be written to.  However, Ext_data and
/// Rt_ext_data use a run-time value (sync) to determine if a block
/// will be written.
template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_from_ptr_impl<Dim, Block, true, false>
{
  typedef Applied_layout<Rt_layout<Dim> > LP;

  typedef Rt_pointer<typename Block::value_type> rt_ptr_type;

  static void copy(Block const*, LP const&, rt_ptr_type)
  {
    assert(false);
  }
};


/// Specialization for blocks with non-complex value_types.
template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_from_ptr_impl<Dim, Block, false, true>
{
  typedef Applied_layout<Rt_layout<Dim> > LP;

  typedef Storage<Cmplx_inter_fmt, typename Block::value_type>
	  inter_storage_type;

  typedef typename inter_storage_type::type inter_ptr_type;
  typedef Rt_pointer<typename Block::value_type> rt_ptr_type;

  static void copy(Block* block, LP const& layout, inter_ptr_type data)
  {
    Length<Dim> ext = extent<Dim>(*block);

    for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
      put(*block, idx, inter_storage_type::get(data, layout.index(idx)));
  }

  static void copy(Block* block, LP const& layout, rt_ptr_type data)
  {
    copy(block, layout, data.as_inter());
  }
};


/// Specialization for const blocks (with non-complex value type).
template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_from_ptr_impl<Dim, Block, false, false>
{
  typedef Applied_layout<Rt_layout<Dim> > LP;

  typedef Rt_pointer<typename Block::value_type> rt_ptr_type;

  static void copy(Block const*, LP const&, rt_ptr_type)
  {
    assert(false);
  }
};



/// Utility classes to copy block data to/from a pointer to regular memory,
/// with layout determined at run-time.
///
/// Template parameters:
///   :Dim: is the dimension of the run-time layout,
///   :Block: is a block type.
template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_to_ptr
  : Rt_block_copy_to_ptr_impl<Dim, Block,
		    vsip::impl::Is_complex<typename Block::value_type>::value>
{};

template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_from_ptr
  : Rt_block_copy_from_ptr_impl<Dim, Block,
		    vsip::impl::Is_complex<typename Block::value_type>::value>
{};




} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_BLOCK_COPY_HPP

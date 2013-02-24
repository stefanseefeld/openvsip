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
///   :D: the dimension
///   :B: the block type
///   :O: the dimension-order
///   :P: packing type
///   :C: complex storage format
template <dimension_type D,
	  typename B,
	  typename O,
	  pack_type P,
	  storage_format_type C>
struct Block_copy_to_ptr
{
  typedef Applied_layout<Layout<D, O, P, C> > layout_type;
  typedef Storage<C, typename B::value_type> storage_type;
  typedef typename storage_type::type ptr_type;


  static void copy (B *block, layout_type &layout, ptr_type data)
  {
    namespace p = vsip::impl::profile;
    p::Scope<p::copy>("Block_copy_to_ptr::copy",
		      block->size() * sizeof(typename B::value_type));
    Length<D> ext = extent<D>(*block);
    expr::evaluate(*block);
    for (Index<D> idx; valid(ext, idx); next(ext, idx))
      storage_type::put(data, layout.index(idx), get(*block, idx));
  }
};

template <dimension_type D,
	  typename B,
	  typename O,
	  pack_type P,
	  storage_format_type C,
	  bool Modifiable = is_modifiable_block<B>::value>
struct Block_copy_from_ptr;

template <dimension_type D,
	  typename B,
	  typename O,
	  pack_type P,
	  storage_format_type C>
struct Block_copy_from_ptr<D, B, O, P, C, true>
{
  typedef Applied_layout<Layout<D, O, P, C> > layout_type;
  typedef Storage<C, typename B::value_type> storage_type;
  typedef typename storage_type::type ptr_type;

  static void copy(B *block, layout_type &layout, ptr_type data)
  {
    namespace p = vsip::impl::profile;
    p::Scope<p::copy>("Block_copy_from_ptr::copy",
		      block->size() * sizeof(typename B::value_type));
    Length<D> ext = extent<D>(*block);

    for (Index<D> idx; valid(ext, idx); next(ext, idx))
    {
      put(*block, idx, storage_type::get(data, layout.index(idx)));
    }
  }
};

template <dimension_type D,
	  typename B,
	  typename O,
	  pack_type P,
	  storage_format_type C>
struct Block_copy_from_ptr<D, B, O, P, C, false>
{
  typedef Applied_layout<Layout<D, O, P, C> > layout_type;
  typedef Storage<C, typename B::value_type> storage_type;
  typedef typename storage_type::type ptr_type;

  static void copy(B *, layout_type &, ptr_type) { assert(0);}
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

  typedef Storage<interleaved_complex, typename Block::value_type>
	  inter_storage_type;
  typedef Storage<split_complex, typename Block::value_type>
	  split_storage_type;

  typedef typename inter_storage_type::type inter_ptr_type;
  typedef typename split_storage_type::type split_ptr_type;

  typedef dda::impl::Pointer<typename Block::value_type> rt_ptr_type;

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
    namespace p = vsip::impl::profile;
    p::Scope<p::copy>("Rt_block_copy_to_ptr::copy",
		      block->size() * sizeof(typename Block::value_type));
    if (layout_storage_format(layout) == interleaved_complex)
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

  typedef Storage<interleaved_complex, typename Block::value_type>
	  inter_storage_type;

  typedef typename inter_storage_type::type inter_ptr_type;
  typedef dda::impl::Pointer<typename Block::value_type> rt_ptr_type;


  static void copy(Block* block, LP const& layout, inter_ptr_type data)
  {
    Length<Dim> ext = extent<Dim>(*block);

    for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
      inter_storage_type::put(data, layout.index(idx), get(*block, idx));
  }

  static void copy(Block* block, LP const& layout, rt_ptr_type data)
  {
    namespace p = vsip::impl::profile;
    p::Scope<p::copy>("Rt_block_copy_to_ptr::copy",
		      block->size() * sizeof(typename Block::value_type));
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
/// error.  This is necessary because dda::Data passes sync as
/// a run-time parameter.
template <dimension_type Dim,
	  typename       Block,
	  bool           IsComplex,
	  bool           Modifiable = is_modifiable_block<Block>::value>
struct Rt_block_copy_from_ptr_impl;

/// Specializations for blocks with complex value_types.
template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_from_ptr_impl<Dim, Block, true, true>
{
  typedef Applied_layout<Rt_layout<Dim> > LP;

  typedef Storage<interleaved_complex, typename Block::value_type>
	  inter_storage_type;
  typedef Storage<split_complex, typename Block::value_type>
	  split_storage_type;

  typedef typename inter_storage_type::type inter_ptr_type;
  typedef typename split_storage_type::type split_ptr_type;

  typedef dda::impl::Pointer<typename Block::value_type> rt_ptr_type;

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
    namespace p = vsip::impl::profile;
    p::Scope<p::copy>("Rt_block_copy_from_ptr::copy",
		      block->size() * sizeof(typename Block::value_type));
    if (layout_storage_format(layout) == interleaved_complex)
      copy(block, layout, data.as_inter());
    else
      copy(block, layout, data.as_split());
  }
};


/// Specialization for const blocks (with complex value type).
///
/// Non-modifiable blocks cannot be written to.  However, dda::Data and
/// Rt_data use a run-time value (sync) to determine if a block
/// will be written.
template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_from_ptr_impl<Dim, Block, true, false>
{
  typedef Applied_layout<Rt_layout<Dim> > LP;

  typedef dda::impl::Pointer<typename Block::value_type> rt_ptr_type;

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

  typedef Storage<interleaved_complex, typename Block::value_type>
	  inter_storage_type;

  typedef typename inter_storage_type::type inter_ptr_type;
  typedef dda::impl::Pointer<typename Block::value_type> rt_ptr_type;

  static void copy(Block* block, LP const& layout, inter_ptr_type data)
  {
    Length<Dim> ext = extent<Dim>(*block);

    for (Index<Dim> idx; valid(ext, idx); next(ext, idx))
      put(*block, idx, inter_storage_type::get(data, layout.index(idx)));
  }

  static void copy(Block* block, LP const& layout, rt_ptr_type data)
  {
    namespace p = vsip::impl::profile;
    p::Scope<p::copy>("Rt_block_copy_from_ptr::copy",
		      block->size() * sizeof(typename Block::value_type));
    copy(block, layout, data.as_inter());
  }
};


/// Specialization for const blocks (with non-complex value type).
template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_from_ptr_impl<Dim, Block, false, false>
{
  typedef Applied_layout<Rt_layout<Dim> > LP;

  typedef dda::impl::Pointer<typename Block::value_type> rt_ptr_type;

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
		    vsip::impl::is_complex<typename Block::value_type>::value>
{};

template <dimension_type Dim,
	  typename       Block>
struct Rt_block_copy_from_ptr
  : Rt_block_copy_from_ptr_impl<Dim, Block,
		    vsip::impl::is_complex<typename Block::value_type>::value>
{};




} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_BLOCK_COPY_HPP

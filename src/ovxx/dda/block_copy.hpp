//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_dda_block_copy_hpp_
#define ovxx_dda_block_copy_hpp_

#include <ovxx/block_traits.hpp>
#include <ovxx/storage.hpp>
#include <ovxx/pointer.hpp>
#include <ovxx/expr.hpp>

namespace ovxx
{
namespace dda
{
namespace detail
{

template <dimension_type D,
	  typename B,
	  typename O,
	  pack_type P,
	  storage_format_type F,
	  bool M = is_modifiable_block<B>::value>
struct Block_copy_from_ptr;

template <dimension_type D,
	  typename B,
	  typename O,
	  pack_type P,
	  storage_format_type F>
struct Block_copy_from_ptr<D, B, O, P, F, true>
{
  typedef Applied_layout<Layout<D, O, P, F> > layout_type;
  typedef storage_traits<typename B::value_type, F> storage;
  typedef typename storage::const_ptr_type ptr_type;

  static void copy(ptr_type data, layout_type const &layout, B &block)
  {
    Length<D> ext = extent<D>(block);
    for (Index<D> idx; valid(ext, idx); next(ext, idx))
      put(block, idx, storage::get(data, layout.index(idx)));
  }
};

template <dimension_type D,
	  typename B,
	  typename O,
	  pack_type P,
	  storage_format_type F>
struct Block_copy_from_ptr<D, B, O, P, F, false>
{
  typedef Applied_layout<Layout<D, O, P, F> > layout_type;
  typedef storage_traits<typename B::value_type, F> storage;
  typedef typename storage::const_ptr_type ptr_type;

  static void copy(ptr_type, layout_type const &, B &) { assert(0);}
};

/// Implementation class to copy block data to pointer to regular memory,
/// with layout determined at run-time.
///
/// Template parameters:
///
///   D: is the dimension of the run-time layout.
///   B: is a block type,
///   C: indicates whether the value type is complex

template <dimension_type D,
	  typename       B,
	  bool           C = is_complex<typename B::value_type>::value>
struct Rt_block_copy_to_ptr;

/// Specialization for blocks with complex value_types.
template <dimension_type D, typename B>
struct Rt_block_copy_to_ptr<D, B, true>
{
  typedef Applied_layout<Rt_layout<D> > layout_type;
  typedef storage_traits<typename B::value_type, array> array_storage;
  typedef storage_traits<typename B::value_type, interleaved_complex> inter_storage;
  typedef storage_traits<typename B::value_type, split_complex> split_storage;

  typedef typename array_storage::ptr_type array_ptr_type;
  typedef typename inter_storage::ptr_type inter_ptr_type;
  typedef typename split_storage::ptr_type split_ptr_type;

  typedef pointer<typename B::value_type> rt_ptr_type;

  static void copy(B const &block, array_ptr_type data, layout_type const &layout)
  {
    Length<D> ext = extent<D>(block);
    for (Index<D> idx; valid(ext, idx); next(ext, idx))
      array_storage::put(data, layout.index(idx), get(block, idx));
  }

  static void copy(B const &block, inter_ptr_type data, layout_type const &layout)
  {
    Length<D> ext = extent<D>(block);
    for (Index<D> idx; valid(ext, idx); next(ext, idx))
      inter_storage::put(data, layout.index(idx), get(block, idx));
  }

  static void copy(B const &block, split_ptr_type data, layout_type const &layout)
  {
    Length<D> ext = extent<D>(block);
    for (Index<D> idx; valid(ext, idx); next(ext, idx))
      split_storage::put(data, layout.index(idx), get(block, idx));
  }

  static void copy(B const &block, rt_ptr_type data, layout_type const &layout)
  {
    switch (layout.storage_format())
    {
      case array:
	copy(block, data.template as<array>(), layout);
	break;
      case interleaved_complex:
	copy(block, data.template as<interleaved_complex>(), layout);
	break;
      case split_complex:
	copy(block, data.template as<split_complex>(), layout);
	break;
      default:
        assert(0);
    }
  }
};

/// Specialization for blocks with non-complex value_types.
template <dimension_type D, typename B>
struct Rt_block_copy_to_ptr<D, B, false>
{
  typedef Applied_layout<Rt_layout<D> > layout_type;
  typedef storage_traits<typename B::value_type, array> storage;
  typedef typename storage::ptr_type ptr_type;
  typedef pointer<typename B::value_type> rt_ptr_type;

  static void copy(B const &block, ptr_type data, layout_type const &layout)
  {
    Length<D> ext = extent<D>(block);
    for (Index<D> idx; valid(ext, idx); next(ext, idx))
      storage::put(data, layout.index(idx), get(block, idx));
  }

  static void copy(B const &block, rt_ptr_type data, layout_type const &layout)
  {
    copy(block, data.template as<array>(), layout);
  }
};

/// Implementation class to copy block data from pointer to regular memory,
/// with layout determined at run-time.
///
/// Template parameters:
///
///   D: is the dimension of the run-time layout.
///   B: is a block type,
template <dimension_type D,
	  typename B,
	  bool C = is_complex<typename B::value_type>::value,
	  bool M = is_modifiable_block<B>::value>
struct Rt_block_copy_from_ptr;

template <dimension_type D, typename B>
struct Rt_block_copy_from_ptr<D, B, true, true>
{
  typedef Applied_layout<Rt_layout<D> > layout_type;
  typedef storage_traits<typename B::value_type, array> array_storage;
  typedef storage_traits<typename B::value_type, interleaved_complex> inter_storage;
  typedef storage_traits<typename B::value_type, split_complex> split_storage;
  typedef typename array_storage::ptr_type array_ptr_type;
  typedef typename inter_storage::ptr_type inter_ptr_type;
  typedef typename split_storage::ptr_type split_ptr_type;
  typedef pointer<typename B::value_type> rt_ptr_type;

  static void copy(array_ptr_type data, layout_type const &layout, B &block)
  {
    Length<D> ext = extent<D>(block);
    for (Index<D> idx; valid(ext, idx); next(ext, idx))
      put(block, idx, array_storage::get(data, layout.index(idx)));
  }
  static void copy(inter_ptr_type data, layout_type const &layout, B &block)
  {
    Length<D> ext = extent<D>(block);
    for (Index<D> idx; valid(ext, idx); next(ext, idx))
      put(block, idx, inter_storage::get(data, layout.index(idx)));
  }
  static void copy(split_ptr_type data, layout_type const &layout, B &block)
  {
    Length<D> ext = extent<D>(block);
    for (Index<D> idx; valid(ext, idx); next(ext, idx))
      put(block, idx, split_storage::get(data, layout.index(idx)));
  }

  static void copy(rt_ptr_type data, layout_type const &layout, B &block)
  {
    switch (layout.storage_format())
    {
      case array:
	copy(data.template as<array>(), layout, block);
	break;
      case interleaved_complex:
	copy(data.template as<interleaved_complex>(), layout, block);
	break;
      case split_complex:
	copy(data.template as<split_complex>(), layout, block);
	break;
      default:
	assert(0);
    }
  }
};

template <dimension_type D, typename B>
struct Rt_block_copy_from_ptr<D, B, true, false>
{
  typedef Applied_layout<Rt_layout<D> > layout_type;
  typedef pointer<typename B::value_type> rt_ptr_type;
  static void copy(rt_ptr_type, layout_type const &, B const &)
  {
    assert(false);
  }
};

template <dimension_type D, typename B>
struct Rt_block_copy_from_ptr<D, B, false, true>
{
  typedef Applied_layout<Rt_layout<D> > layout_type;
  typedef storage_traits<typename B::value_type, array> storage;
  typedef typename storage::ptr_type ptr_type;
  typedef pointer<typename B::value_type> rt_ptr_type;

  static void copy(ptr_type data, layout_type const &layout, B &block)
  {
    Length<D> ext = extent<D>(block);
    for (Index<D> idx; valid(ext, idx); next(ext, idx))
      put(block, idx, storage::get(data, layout.index(idx)));
  }

  static void copy(rt_ptr_type data, layout_type const &layout, B &block)
  {
    copy(data.template as<array>(), layout, block);
  }
};

template <dimension_type D, typename B>
struct Rt_block_copy_from_ptr<D, B, false, false>
{
  typedef Applied_layout<Rt_layout<D> > layout_type;
  typedef typename B::value_type *ptr_type;
  static void copy(ptr_type, layout_type const &, B const &)
  { assert(false);}
};

} // namespace ovxx::dda::detail

/// copy block data to regular memory according to the given layout.
template <dimension_type D, typename B, typename O,
	  pack_type P, storage_format_type F>
void block_copy(B const &block,
		typename storage_traits<typename B::value_type, F>::ptr_type data,
		Applied_layout<Layout<D, O, P, F> > const &layout)
{
  typedef storage_traits<typename B::value_type, F> storage;
  Length<D> ext = extent<D>(block);
  expr::evaluate(block);
  for (Index<D> idx; valid(ext, idx); next(ext, idx))
    storage::put(data, layout.index(idx), get(block, idx));
};

template <dimension_type D, typename B>
void block_copy(B const &block,
		typename B::value_type *data,
		Applied_layout<Rt_layout<D> > const &layout)
{
  detail::Rt_block_copy_to_ptr<D, B>::copy(block, data, layout);
}

template <dimension_type D, typename B>
void block_copy(B const &block,
		pointer<typename B::value_type> data,
		Applied_layout<Rt_layout<D> > const &layout)
{
  detail::Rt_block_copy_to_ptr<D, B>::copy(block, data, layout);
}

/// copy data from regular memory, described by the given layout into a block.
template <dimension_type D, typename B, typename O,
	  pack_type P, storage_format_type F>
void block_copy(typename storage_traits<typename B::value_type, F>::const_ptr_type data,
		Applied_layout<Layout<D, O, P, F> > const &layout,
		B &block)
{
  detail::Block_copy_from_ptr<D, B, O, P, F>::copy(data, layout, block);
}

template <dimension_type D, typename B>
void block_copy(typename B::value_type *data,
		Applied_layout<Rt_layout<D> > const &layout,
		B &block)
{
  detail::Rt_block_copy_from_ptr<D, B>::copy(data, layout, block);
};

template <dimension_type D, typename B>
void block_copy(pointer<typename B::value_type> data,
		Applied_layout<Rt_layout<D> > const &layout,
		B &block)
{
  detail::Rt_block_copy_from_ptr<D, B>::copy(data, layout, block);
};

} // namespace ovxx::dda
} // namespace ovxx

#endif

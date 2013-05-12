//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

/// Description:
///   VSIPL++ Library: Data layout within a block.

#ifndef vsip_layout_hpp_
#define vsip_layout_hpp_

#include <vsip/support.hpp>
#include <vsip/impl/complex_decl.hpp>
#include <vsip/domain.hpp>

namespace vsip
{
enum pack_type
{
  /// no packing information is available
  any_packing = 0,
  /// data has unit-stride
  unit_stride = 1,
  /// data is contiguous
  dense = 2,
  /// data has unit-stride in minor dimension,
  /// and rows / columns / planes are known 
  /// to start on aligned boundaries.
  aligned = 7,
  aligned_8 = 8,
  aligned_16 = 16,
  aligned_32 = 32,
  aligned_64 = 64,
  aligned_128 = 128,
  aligned_256 = 256,
  aligned_512 = 512,
  aligned_1024 = 1024
};

template <pack_type>
struct is_packing_unit_stride { static bool const value = false;};

#define VSIP_IMPL_UNIT_STRIDE(P)            \
template <>				    \
struct is_packing_unit_stride<P>	    \
{ static bool const value = true;};

VSIP_IMPL_UNIT_STRIDE(unit_stride)
VSIP_IMPL_UNIT_STRIDE(dense)
VSIP_IMPL_UNIT_STRIDE(aligned)
VSIP_IMPL_UNIT_STRIDE(aligned_8)
VSIP_IMPL_UNIT_STRIDE(aligned_16)
VSIP_IMPL_UNIT_STRIDE(aligned_32)
VSIP_IMPL_UNIT_STRIDE(aligned_64)
VSIP_IMPL_UNIT_STRIDE(aligned_128)
VSIP_IMPL_UNIT_STRIDE(aligned_256)
VSIP_IMPL_UNIT_STRIDE(aligned_512)
VSIP_IMPL_UNIT_STRIDE(aligned_1024)

#undef VSIP_IMPL_UNIT_STRIDE

enum storage_format_type
{
  /// storage format is unknown.
  any_storage_format = 0,
  /// data is stored as a simple array.
  array, 
  /// complex data is stored in separate arrays of real and imaginary data.
  split_complex,
  /// complex data is stored as an array of real data, with interleaved
  /// real and imaginary parts.
  interleaved_complex
};

/// Layout groups a block's (compile-time) data layout information.
///
/// Template parameters:
///   :D: The block's dimension.
///   :Order: The dimension order (`tuple<0, 1, 2>`, `row2_type`, etc.)
///   :packing: The pack-type of the layout.
///              `packing::aligned` <...>.
///   :storage_format: The storage-format.
template <dimension_type D,
	  typename Order,
	  pack_type P,
	  storage_format_type S = array>
struct Layout
{
  static dimension_type const dim = D;
  typedef Order order_type;
  static pack_type const packing = P;
  static storage_format_type const storage_format = S;
};

template <typename Block>
struct get_block_layout
{
  static dimension_type const dim = Block::dim;
  typedef tuple<0, 1, 2>   order_type;
  static pack_type const packing = any_packing;
  static storage_format_type const storage_format = array;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <typename B> 
struct get_block_layout<B const> : get_block_layout<B> {};

template <typename Block>
struct supports_dda { static bool const value = false;};

template <typename B>
struct supports_dda<B const> : supports_dda<B> {};

} // namespace vsip

#endif

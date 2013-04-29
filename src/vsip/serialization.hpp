//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_serialization_hpp_
#define vsip_serialization_hpp_

#include <vsip/dda.hpp>
#include <vsip/core/inttypes.hpp>
#include <vsip/dense.hpp>
#include <cstring>

namespace vsip
{
namespace serialization
{

struct Descriptor
{
  char value_type[8];
  vsip::impl::uint8_type dimensions;
  vsip::impl::uint8_type storage_format;
  vsip::impl::uint64_type size[3];
  vsip::impl::uint64_type stride[3];
};

/// type_info is a helper to map between C++ types
/// and type-encodings in a Descriptor.
template <typename T>
struct type_info
{
  /// Fill out d.value_type from 'T'
  static void write(Descriptor &d);
  /// Report whether d.value_type matches 'T'.
  static bool check(Descriptor const &);
};

#define VSIP_IMPL_TYPE_INFO(T, N)		\
template <>					\
struct type_info<T>				\
{						\
  static void write(Descriptor &d)		\
  { strcpy(d.value_type, N);}			\
  static bool check(Descriptor const &d)	\
  { return strcmp(d.value_type, N) == 0;}	\
};

VSIP_IMPL_TYPE_INFO(char, "c")
VSIP_IMPL_TYPE_INFO(short, "s")
VSIP_IMPL_TYPE_INFO(int, "i")
VSIP_IMPL_TYPE_INFO(float, "f")
VSIP_IMPL_TYPE_INFO(double, "d")
VSIP_IMPL_TYPE_INFO(complex<float>, "cf")
VSIP_IMPL_TYPE_INFO(complex<double>, "cd")

#undef VSIP_IMPL_TYPE_INFO

namespace impl
{

template <typename B>
struct block_compatibility
{
  static bool check(Descriptor const &) { return false;}
};

template <typename O>
struct ordering_compatibility;

template <dimension_type D0, dimension_type D1, dimension_type D2>
struct ordering_compatibility<tuple<D0, D1,D2> >
{
  static bool check(Descriptor const &info)
  {
    // The minor dimension needs to be unit-stride.
    if (info.dimensions == 1 && info.stride[D0] != 1) return false;
    else if (info.dimensions == 2 && info.stride[D1] != 1) return false;
    else if (info.dimensions == 3 && info.stride[D2] != 1) return false;
    // Make sure strides match sizes.
    if (info.dimensions == 2 && info.stride[D0] != info.size[D1]) return false;
    else if (info.dimensions == 3 && info.stride[D0] != info.size[D2] * info.size[D1]) return false;
    return true;
  }
};

template <dimension_type D, typename T, typename O, storage_format_type S>
struct block_compatibility<vsip::impl::Strided<D, T, Layout<D, O, dense, S>, Local_map> >
{
  static bool check(Descriptor const &info)
  {
    if (!type_info<T>::check(info))
      return false;
    else if (D != info.dimensions)
      return false; // dimension mismatch
    
    // TODO: support non-dense layouts
    if (!ordering_compatibility<O>::check(info))
      return false;
    return true;
  }
};

template <dimension_type D, typename T, typename O>
struct block_compatibility<Dense<D, T, O, Local_map> > : block_compatibility<
  vsip::impl::Strided<D, T,
    Layout<D, O, dense, Dense<D, T, O, Local_map>::storage_format>, Local_map> >
{};

} // vsip::serialization::impl


/// Fill in a descriptor from a DDA object. This makes it possible
/// to serialize block data by using a dda::Data proxy.
///
/// Arguments:
///   :data: The Data object from which to extract the information.
///   :desc: The Descriptor object to fill out.
template <typename B, vsip::dda::sync_policy S, typename L>
void describe_data(vsip::dda::Data<B, S, L> const &data, Descriptor &info)
{
  type_info<typename B::value_type>::write(info);
  info.dimensions = B::dim;
  if (vsip::impl::is_split_block<B>::value)
    info.storage_format = split_complex;
  else
    info.storage_format = interleaved_complex;
  info.size[0] = data.size(0);
  info.stride[0] = data.stride(0);
  if (info.dimensions > 1)
  {
    info.size[1] = data.size(1);
    info.stride[1] = data.stride(1);
  }
  if (info.dimensions > 2)
  {
    info.size[2] = data.size(2);
    info.stride[2] = data.stride(2);
  }
}

/// Fill in a descriptor from a block's user storage.
/// This is only valid if `B` supports user-storage.
///
/// Arguments:
///   :block: The block holding user-storage.
///   :desc: The Descriptor object to fill out.
template <typename B>
void describe_user_storage(B const &block, Descriptor &info)
{
  type_info<typename B::value_type>::write(info);
  info.dimensions = B::dim;
  info.storage_format = block.user_storage();
  
  vsip::impl::Applied_layout<typename get_block_layout<B>::type> layout
    (vsip::impl::extent<B::dim>(block));

  switch (info.dimensions)
  {
    case 1:
      info.size[0] = block.size();
      info.size[1] = info.size[2] = 0;
      info.stride[0] = 1;
      info.stride[1] = info.stride[2] = 0;
      break;
    case 2:
      info.size[0] = block.size(2, 0);
      info.size[1] = block.size(2, 1);
      info.size[2] = 0;
      info.stride[0] = layout.stride(0);
      info.stride[1] = layout.stride(1);
      info.stride[2] = 0;
      break;
    case 3:
      info.size[0] = block.size(3, 0);
      info.size[1] = block.size(3, 1);
      info.size[2] = block.size(3, 2);
      info.stride[0] = layout.stride(0);
      info.stride[1] = layout.stride(1);
      info.stride[2] = layout.stride(2);
      break;
  }
}

/// Report whether the given block type is compatible with
/// the data described in `info`.
/// If it is, `b.rebind(ptr)` is a valid expression for
/// `b` being of type `B` and `ptr` being a raw pointer matching
/// the `info` descriptor.
template <typename B>
bool is_compatible(Descriptor const &info) 
{ return impl::block_compatibility<B>::check(info);}

} // namespace vsip::serialization
} // namespace vsip

#endif

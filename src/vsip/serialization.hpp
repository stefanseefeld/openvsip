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
#include <vsip/dense.hpp>
#include <ovxx/inttypes.hpp>
#include <cstring>

namespace vsip
{
namespace serialization
{
typedef ovxx::uint8_type uint8_type;
typedef ovxx::int8_type int8_type;
typedef ovxx::uint16_type uint16_type;
typedef ovxx::int16_type int16_type;
typedef ovxx::uint32_type uint32_type;
typedef ovxx::int32_type int32_type;
typedef ovxx::uint64_type uint64_type;
typedef ovxx::int64_type int64_type;

struct Descriptor
{
  uint64_type value_type;
  uint8_type dimensions;
  uint8_type storage_format;
  uint64_type size[3];
  int64_type stride[3];
  uint64_type storage_size;
};

namespace impl
{
// The following helper code allows to establish type equivalence,
// to match integral types across architectures.
template <typename T,
	  uint64_type S = sizeof(T),
	  uint64_type U = ovxx::is_unsigned<T>::value>
struct integral_type_traits
{
  static uint64_type const value =  (S << 10) + (U << 9) + (1UL << 8);
};

inline bool is_integral(uint64_type v) { return v & (1UL << 8);}
inline bool is_unsigned(uint64_type v) { return v & (1UL << 9);}
inline unsigned size(uint64_type v) { return v >> 10;}
}

/// type_info is a helper to map between C++ types
/// and type-encodings in a Descriptor.
template <typename T> struct type_info;

#define OVXX_TYPE_INFO(T, N)			\
template <>					\
struct type_info<T>				\
{						\
  static uint64_type const value = N;		\
};

// The first 8 bits are used for builtin non-integral types
OVXX_TYPE_INFO(bool, 0)
OVXX_TYPE_INFO(char, 1)
OVXX_TYPE_INFO(float, 4)
OVXX_TYPE_INFO(complex<float>, 5)
OVXX_TYPE_INFO(double, 6)
OVXX_TYPE_INFO(complex<double>, 7)

#undef OVXX_TYPE_INFO

#define OVXX_INT_TYPE_INFO(T)			\
template <>					\
struct type_info<T>				\
{						\
  static uint64_type const value =		\
    impl::integral_type_traits<T>::value;	\
};

// The second 8 bits are used for builtin integral types
// (Note: "integral types" in this context is referring to
//        the C99 extended integral types, and specifically
//        excludes 'bool' and 'char'.)
OVXX_INT_TYPE_INFO(int8_type)
OVXX_INT_TYPE_INFO(uint8_type)
OVXX_INT_TYPE_INFO(int16_type)
OVXX_INT_TYPE_INFO(uint16_type)
OVXX_INT_TYPE_INFO(int32_type)
OVXX_INT_TYPE_INFO(uint32_type)
OVXX_INT_TYPE_INFO(int64_type)
OVXX_INT_TYPE_INFO(uint64_type)

#undef OVXX_INT_TYPE_INFO

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
struct ordering_compatibility<tuple<D0, D1, D2> >
{
  static bool check(Descriptor const &info)
  {
    // The minor dimension needs to be unit-stride.
    if (info.dimensions == 1 && info.stride[D0] != 1) return false;
    else if (info.dimensions == 2 && info.stride[D1] != 1) return false;
    else if (info.dimensions == 3 && info.stride[D2] != 1) return false;
    // Make sure strides match sizes.
    if (info.dimensions == 2 && 
	(info.stride[D0] < 0 ||
	 static_cast<length_type>(info.stride[D0]) != info.size[D1])) return false;
    else if (info.dimensions == 3 && 
	     (info.stride[D0] < 0 ||
	      static_cast<length_type>(info.stride[D0]) != info.size[D2] * info.size[D1])) return false;
    return true;
  }
};

template <dimension_type D, typename T, typename O, storage_format_type S>
struct block_compatibility<ovxx::Strided<D, T, Layout<D, O, dense, S>, Local_map> >
{
  static bool check(Descriptor const &info)
  {
    // If this is an integral type, check for type equivalence (size, signedness)...
    if (is_integral(type_info<T>::value))
    {
      if (!is_integral(info.value_type)) return false;
      else if (is_unsigned(type_info<T>::value) != is_unsigned(info.value_type)) return false;
      else if (size(type_info<T>::value) != size(info.value_type)) return false;
    }
    // ...else we require strict type equality
    else if (type_info<T>::value != info.value_type)
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
  ovxx::Strided<D, T,
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
  info.value_type = type_info<typename B::value_type>::value;
  info.dimensions = B::dim;
  info.storage_format = vsip::dda::Data<B, S, L>::layout_type::storage_format;
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
  info.storage_size = data.storage_size();
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
  info.value_type = type_info<typename B::value_type>::value;
  info.dimensions = B::dim;
  info.storage_format = block.user_storage();
  
  ovxx::Applied_layout<typename get_block_layout<B>::type> layout
    (ovxx::extent<B::dim>(block));

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
  info.storage_size = block.size();
  if (info.storage_format != array) info.storage_size *= 2;
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

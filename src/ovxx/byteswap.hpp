//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_byteswap_hpp_
#define ovxx_byteswap_hpp_

#include <vsip/complex.hpp>
#include <ovxx/detail/endian.hpp>
#include <byteswap.h>

namespace ovxx
{
namespace detail
{
template <typename T = char,
	  bool to_swap_or_not_to_swap = false,
	  std::size_t type_size = sizeof(T),
	  bool IsComplex = is_complex<T>::value>
struct bswap 
{
  static void apply(T *) {}
};

template <typename T>
struct bswap<T, true, 2, false>
{
  static void apply(T *d) { bswap_16(reinterpret_cast<char*>(d));}
};

template <typename T>
struct bswap<T, true, 4, false>
{
  static void apply(T *d) { bswap_32(reinterpret_cast<char*>(d));}
};

template <typename T>
struct bswap<T, true, 8, false>
{
  static void apply(T *d) { bswap_64(reinterpret_cast<char*>(d));}
};

template <typename T>
struct bswap<T, true, 8, true>   // complex
{
  static void apply(T *d) 
  {
    bswap_32(reinterpret_cast<char*>(d));
    bswap_32(reinterpret_cast<char*>(d)+4);
  }
};

template <typename T>
struct bswap<T, true, 16, true>  // complex
{
  static void apply(T *d) 
  {
    bswap_64(reinterpret_cast<char*>(d));
    bswap_64(reinterpret_cast<char*>(d)+8);
  }
};

} // namespace ovxx::detail

// swap bytes
template <typename T>
void byteswap(T &data, bool swap_bytes = true)
{
  if(swap_bytes) detail::bswap<T,true>::apply(&data);
}

// swap bytes, if 'C == true'
template <bool C, typename T>
void byteswap(T &data)
{
  detail::bswap<T, C>::apply(&data);
}

// swap bytes unconditionally
template <typename T>
void byteswap(T &data)
{
  detail::bswap<T, true>::apply(&data);
}


} // namespace ovxx

#endif

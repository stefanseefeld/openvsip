/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ppu/util.hpp
    @author  Jules Bergmann
    @date    2007-04-13
    @brief   VSIPL++ Library: Utilities for the IBM Cell/B.E.
*/

#ifndef VSIP_OPT_CBE_PPU_UTIL_HPP
#define VSIP_OPT_CBE_PPU_UTIL_HPP

/// DMA starting address alignment (in bytes).
#define VSIP_IMPL_CBE_DMA_ALIGNMENT 16

/// Bulk DMA size granularity (in bytes) 
///
/// DMAs larger than 16 bytes must have a granularity of 16 bytes.
/// Small DMAs of fixed size 1, 2, 4, and 8 bytes are also allowed.
#define VSIP_IMPL_CBE_DMA_GRANULARITY 16

namespace vsip
{
namespace impl
{
namespace cbe
{

/// Determine if DMA size (in bytes) is valid for a bulk DMA.
inline bool
is_dma_size_ok(length_type size_in_bytes)
{
  return (size_in_bytes == 1 ||
	  size_in_bytes == 2 ||
	  size_in_bytes == 4 ||
	  size_in_bytes == 8 ||
	  size_in_bytes % 16 == 0);
}


/// Determine if DMA address is properly aligned.
template <typename T>
inline bool
is_dma_addr_ok(T const* addr)
{
  return ((intptr_t)addr & (VSIP_IMPL_CBE_DMA_ALIGNMENT - 1)) == 0;
}



template <typename T>
inline bool
is_dma_addr_ok(std::pair<T*, T*> const& addr)
{
  return is_dma_addr_ok(addr.first) && is_dma_addr_ok(addr.second);
}


/// Determine if stride will cause an unaligned DMA
template <typename T>
inline bool
is_dma_stride_ok(stride_type stride)
{
  return ((stride * sizeof(T)) & (VSIP_IMPL_CBE_DMA_GRANULARITY - 1)) == 0;
}




/// Convert a pointer into a 64-bit effective address (EA).
///
/// Note: just casting a 32-bit pointer to an unsigned long long does
///
///   unsigned long long ea = reinterpret_cast<unsigned long long>(ptr);
///
/// does not work.  If the high-order bit of the pointer is set, it will
/// be sign extended into the upper 32-bits of the long long.  Using such
/// an errant EA with mfc_get fails (but curiously it works with ALF's
/// ALF_DT_LIST_ADD_ENTRY).
template <typename T>
inline
unsigned long long
ea_from_ptr(T* ptr)
{
  if (sizeof(T*) == sizeof(unsigned long long))
    return reinterpret_cast<unsigned long long>(ptr);
  else
  {
    union
    {
      unsigned long      ptr32[2];
      unsigned long long ptr64;
    } u;
    u.ptr32[0] = 0;
    u.ptr32[1] = reinterpret_cast<unsigned long>(ptr);
    return u.ptr64;
  }
}

/// Size threshold for elementwise operations.
template <typename Operation, bool Split = false>
struct Size_threshold
{
  static int const value = 0;
};

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_CBE_PPU_UTIL_HPP

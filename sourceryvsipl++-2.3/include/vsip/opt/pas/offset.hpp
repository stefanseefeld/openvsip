/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/pas/offset.hpp
    @author  Jules Bergmann
    @date    2006-09-01
    @brief   VSIPL++ Library: Offset class.
*/

#ifndef VSIP_OPT_PAS_OFFSET_HPP
#define VSIP_OPT_PAS_OFFSET_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/layout.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

template <typename ComplexFmt,
	  typename T>
struct Offset
{
  typedef stride_type type;

  static type create(length_type)
  { return type(0); }

  static type offset(type orig, stride_type delta)
  { return orig + delta; }

  static void check_imag_offset(length_type, stride_type imag_offset)
  {
    assert(imag_offset == 0);
  }
};



template <typename T>
struct Offset<Cmplx_split_fmt, complex<T> >
{
  typedef std::pair<stride_type, stride_type> type;

  static type create(length_type size)
  {
    // Size of scalar_type must evenly divide the alignment.
    assert(VSIP_IMPL_PAS_ALIGNMENT % sizeof(T) == 0);

    // Compute the padding and expected offset.
    size_t t_alignment = (VSIP_IMPL_PAS_ALIGNMENT / sizeof(T));
    size_t offset      = size;
    size_t extra       = offset % t_alignment;

    // If not naturally aligned (extra != 0), pad by t_alignment - extra.
    if (extra) offset += (t_alignment - extra);
    return type(0, offset);
  }

  static type offset(type orig, stride_type delta)
  { return type(orig.first + delta, orig.second + delta); }

  static void check_imag_offset(length_type size, stride_type imag_offset)
  {
    type ref = create(size);
    assert(ref.second == imag_offset);
  }
};

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_PAS_OFFSET_HPP

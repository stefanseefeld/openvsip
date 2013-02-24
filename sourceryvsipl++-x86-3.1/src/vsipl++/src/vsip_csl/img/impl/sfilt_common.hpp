/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/img/impl/sfilt_common.hpp
    @author  Jules Bergmann
    @date    2007-10-05
    @brief   VSIPL++ Library: Generic separable filter.
*/

#ifndef VSIP_CSL_IMG_IMPL_SFILT_COMMON_HPP
#define VSIP_CSL_IMG_IMPL_SFILT_COMMON_HPP

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/signal/types.hpp>

namespace vsip_csl
{
namespace img
{

enum edge_handling_type
{
  edge_zero,
  edge_mirror,
  edge_wrap,
  edge_scale
};


namespace impl
{

inline vsip::Domain<2>
sfilt_output_size(
  vsip::support_region_type supp_ct,
  vsip::Domain<2>           kernel_size,
  vsip::Domain<2>           input_size)
{
  (void)kernel_size;
  if (supp_ct == vsip::support_min_zeropad)
    return input_size;
  else
    VSIP_IMPL_THROW(std::runtime_error(
      "sfilt_output_size: support not implemented"));
}



// Separable filter, minimum support size, zero-pad output
//
// Operations:
//   rows * (cols - nk0 + 1) * 2 * nk0 +
//   cols * (rows - nk1 + 1) * 2 * nk1

template <typename T>
inline void
sfilt_min_zeropad(
  T const*          k0,		// coeff vector for 0-dim (stride 1)
  vsip::length_type nk0,	// number of k0's
  T const*          k1,		// coeff vector for 1-dim (stride 1)
  vsip::length_type nk1,	// number of k1's

  T const*          in,
  vsip::stride_type in_row_stride,
  vsip::stride_type in_col_stride ATTRIBUTE_UNUSED,

  T*          	    out,
  vsip::stride_type out_row_stride,
  vsip::stride_type out_col_stride,

  T*                tmp,		// rows * cols

  vsip::length_type rows,		// Pr
  vsip::length_type cols)		// Pc
{
  assert(in_col_stride == 1);
  using vsip::index_type;
  using vsip::length_type;

  length_type b_nk0 = (nk0)/2, e_nk0 = nk0 - b_nk0 - 1;
  length_type b_nk1 = (nk1)/2, e_nk1 = nk1 - b_nk1 - 1;

  T* ptmp = tmp;

  // Filter horizontally along rows.
  for (index_type r=0; r < rows; r++)
  {
    // 1D convolution
    for (index_type c=b_nk1; c<cols-e_nk1; c++)  
    {
      T sum = T();
      index_type i = c-b_nk1;
      for (index_type j=0; j<nk1; j++)
	sum += in[i+j] * k1[j];
      tmp[c] = sum;
    }

    // Advance to next row.
    tmp += cols;
    in  += in_row_stride;	     
  }

  tmp = ptmp;

  // Filter vertically along columns.

  // Zero Top
  for(index_type r = 0; r < b_nk0; r++)  
    for(index_type c = 0; c < cols; c++)
      out[r*out_row_stride + c] = 0;

  // Zero Bottom
  for(index_type r = rows-e_nk0; r < rows; r++)  
    for(index_type c = 0; c < cols; c++)
      out[r*out_row_stride+c] = 0;

  // Zero LHS
  for(index_type c = 0; c < b_nk1; c++)
  {
    for(index_type r = b_nk0; r < rows-e_nk0; r++)  
      out[r*out_row_stride] = 0;
    // Advance to next column.
    tmp += 1;
    out += out_col_stride;	  	  
  }

  for(index_type c = b_nk1; c < cols-e_nk1; c++)
  {
    // 1D convolution
    for (index_type r = b_nk0; r<rows-e_nk0; r++)  
    {
      T sum = T();
      index_type i = (r-b_nk0)*cols;
      
      for (index_type j=0; j<nk0; j++)
	sum += tmp[i+j*cols]*k0[j];
      out[r*out_row_stride] = sum;            
    }

    // Advance to next column.
    tmp += 1;
    out += out_col_stride;	  	  
  }

  // Zero RHS
  for(index_type c = cols-e_nk1; c < cols; c++)
  {
    for(index_type r = b_nk0; r < rows-e_nk0; r++)  
      out[r*out_row_stride] = 0;

    // Advance to next column.
    out += out_col_stride;	  	  
  }
}



} // namespace vsip_csl::img::impl
} // namespace vsip_csl::img

namespace dispatcher
{
namespace op
{
template <dimension_type D,
	  vsip::support_region_type R,
	  img::edge_handling_type E,
	  unsigned N,
          vsip::alg_hint_type H>
struct sfilt;

} // namespace vsip_csl::dispatcher::op
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_CSL_IMG_IMPL_SFILT_COMMON_HPP

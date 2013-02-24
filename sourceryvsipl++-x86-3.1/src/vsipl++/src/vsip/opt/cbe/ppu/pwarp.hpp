/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ppu/pwarp.hpp
    @author  Jules Bergmann
    @date    2007-11-19
    @brief   VSIPL++ Library: Perspective warp bridge with the CBE ALF.
*/

#ifndef VSIP_OPT_CBE_PPU_PWARP_HPP
#define VSIP_OPT_CBE_PPU_PWARP_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/opt/cbe/pwarp_params.h>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/matrix.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cbe
{

// Maximum number of columns that CBE pwarp can handle.

length_type const pwarp_block_max_col_size = 4096;

// Foward decl: ALF bridge function for perspective warp.

template <typename T>
void
pwarp_block_impl(
  Matrix<float> P,
  T const*      p_in,
  stride_type   in_stride_0,
  T*            p_out,
  stride_type   out_stride_0,
  length_type   in_rows,
  length_type   in_cols,
  length_type   out_rows,
  length_type   out_cols);



template <typename CoeffT,
	  typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
pwarp_block(
  const_Matrix<CoeffT, Block1> P,
  const_Matrix<T, Block2>      in,
  Matrix<T, Block3>            out)
{
  length_type out_rows = out.size(0);
  length_type out_cols = out.size(1);
  length_type in_rows  = in.size(0);
  length_type in_cols  = in.size(1);

  dda::Data<Block2, dda::in> data_in(in.block());
  dda::Data<Block3, dda::out> data_out(out.block());

  pwarp_block_impl(
    P,
    data_in.ptr(),  data_in.stride(0),
    data_out.ptr(), data_out.stride(0),
    in_rows, in_cols,
    out_rows, out_cols);
}


} // namespace vsip::impl::fftm
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_CBE_PPU_PWARP_HPP

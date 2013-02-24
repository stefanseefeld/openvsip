/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ukernel_params.h
    @author  Jules Bergmann
    @date    2008-01-23
    @brief   VSIPL++ Library: Parameters for Ukernel.
*/

#ifndef VSIP_OPT_CBE_UKERNEL_PARAMS_HPP
#define VSIP_OPT_CBE_UKERNEL_PARAMS_HPP

/***********************************************************************
  Definitions
***********************************************************************/

#ifdef _cplusplus
namespace vsip
{
namespace impl
{
namespace cbe
{
#endif

/// Structures used in DMAs should be sized in multiples of 128-bits
struct Uk_stream
{
  unsigned long long addr;
  unsigned long long addr_split;
  unsigned int       chunk_offset;
  unsigned int       dim;
  unsigned int       chunk_size0;
  unsigned int       chunk_size1;
  unsigned int       chunk_size2;
  unsigned int       chunk_size0_extra;
  unsigned int       chunk_size1_extra;
  unsigned int       chunk_size2_extra;
  unsigned int       stride0;
  unsigned int       stride1;
  unsigned int       stride2;
  unsigned int       num_chunks0;
  unsigned int       num_chunks1;
  unsigned int       num_chunks2;
  unsigned short     leading_overlap0;
  unsigned short     leading_overlap1;
  unsigned short     leading_overlap2;
  unsigned short     trailing_overlap0;
  unsigned short     trailing_overlap1;
  unsigned short     trailing_overlap2;
  unsigned char       skip_first_overlap0;
  unsigned char       skip_first_overlap1;
  unsigned char       skip_first_overlap2;
  unsigned char       skip_last_overlap0;
  unsigned char       skip_last_overlap1;
  unsigned char       skip_last_overlap2;
  unsigned int       align_shift;
};

template <unsigned int PreArgs,
	  unsigned int InArgs,
	  unsigned int OutArgs,
	  typename     ParamT>
struct Ukernel_params
{
  unsigned long long code_ea;
  int                code_size;
  int                cmd;

  int       rank;
  int       nspe;
  Uk_stream in_stream[PreArgs+InArgs];
  Uk_stream out_stream[OutArgs];
  int       pre_chunks;
  ParamT    kernel_params;
};

struct Empty_params {};

#ifdef _cplusplus
} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
#endif

#endif

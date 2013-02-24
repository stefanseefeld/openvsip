/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/vmmul_params.h
    @author  Don McCoy
    @date    2007-03-14
    @brief   VSIPL++ Library: Parameters for vmmul_c kernel.
*/

#ifndef VSIP_OPT_CBE_VMMUL_PARAMS_H
#define VSIP_OPT_CBE_VMMUL_PARAMS_H

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


#ifndef VSIP_IMPL_MIN_VMMUL_SIZE
#define VSIP_IMPL_MIN_VMMUL_SIZE	  4
#endif

#ifndef VSIP_IMPL_MAX_VMMUL_SIZE
#define VSIP_IMPL_MAX_VMMUL_SIZE	  4096
#endif



typedef struct
{
  int                cmd;

  unsigned int       length;
  unsigned int       input_stride;
  unsigned int       output_stride;
  unsigned int       shift;
  unsigned int       mult;

  unsigned long long ea_input_vector;
  unsigned long long ea_input_matrix;

  unsigned long long ea_output_matrix;
} Vmmul_params;

typedef struct
{
  int                cmd;

  unsigned int length;
  unsigned int input_stride;
  unsigned int output_stride;
  unsigned int shift;
  unsigned int mult;

  unsigned long long ea_input_vector_re;
  unsigned long long ea_input_vector_im;
  unsigned long long ea_input_matrix_re;
  unsigned long long ea_input_matrix_im;

  unsigned long long ea_output_matrix_re;
  unsigned long long ea_output_matrix_im;
} Vmmul_split_params;


#ifdef _cplusplus
} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
#endif

#endif // VSIP_OPT_CBE_VMMUL_PARAMS_H

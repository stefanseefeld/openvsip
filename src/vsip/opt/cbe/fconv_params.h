/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/fconv_params.h
    @author  Don McCoy
    @date    2007-02-04
    @brief   VSIPL++ Library: Parameters for fast convolution kernels.
*/

#ifndef VSIP_OPT_CBE_FCONV_PARAMS_H
#define VSIP_OPT_CBE_FCONV_PARAMS_H

/***********************************************************************
  Definitions
***********************************************************************/

// Fast convolution shares the same minimum as FFT, but the maximum
// is less as more memory is required.
#ifndef VSIP_IMPL_MIN_FCONV_SIZE
#define VSIP_IMPL_MIN_FCONV_SIZE	32
#endif

#ifndef VSIP_IMPL_MAX_FCONV_SIZE
#define VSIP_IMPL_MAX_FCONV_SIZE	4096
#endif

#ifndef VSIP_IMPL_MIN_FCONV_SPLIT_SIZE
#define VSIP_IMPL_MIN_FCONV_SPLIT_SIZE	64
#endif

#ifndef VSIP_IMPL_MAX_FCONV_SPLIT_SIZE
#define VSIP_IMPL_MAX_FCONV_SPLIT_SIZE	4096
#endif



typedef struct
{
  unsigned int       instance_id;
  unsigned int       elements;
  unsigned int       transform_kernel;

  unsigned long long ea_kernel;

  unsigned long long ea_input;
  unsigned long long ea_output;

  unsigned int       kernel_stride;
  unsigned int       input_stride;
  unsigned int       output_stride;
} Fastconv_params;

typedef struct
{
  unsigned int       instance_id;
  unsigned int       elements;
  unsigned int       transform_kernel;

  unsigned long long ea_kernel_re;
  unsigned long long ea_kernel_im;

  unsigned long long ea_input_re;
  unsigned long long ea_input_im;

  unsigned long long ea_output_re;
  unsigned long long ea_output_im;

  unsigned int       kernel_stride;
  unsigned int       input_stride;
  unsigned int       output_stride;
} Fastconv_split_params;


#endif // VSIP_OPT_CBE_FCONV_PARAMS_H

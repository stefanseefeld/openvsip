/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/fft_params.h
    @author  Don McCoy
    @date    2007-03-27
    @brief   VSIPL++ Library: Parameters for FFT kernels.
*/

#ifndef VSIP_OPT_CBE_FFT_PARAMS_H
#define VSIP_OPT_CBE_FFT_PARAMS_H

/***********************************************************************
  Definitions
***********************************************************************/

// Note: the minimum size is determined by the fact that the SPE
// algorithm hand unrolls one loop, doubling the minimum of 16.
#ifndef MIN_FFT_1D_SIZE
#define MIN_FFT_1D_SIZE	  4
#endif

// The maximum size may be up to, but no greater than 8K due to the
// internal memory requirements of the algorithm.  This is further 
// limited here to allow more headroom for fast convolution.
#ifndef MAX_FFT_1D_SIZE
#define MAX_FFT_1D_SIZE	  8192
#endif



typedef enum
{
  fwd_fft = 0,
  inv_fft
} fft_dir_type;


// Structures used in DMAs should be sized in multiples of 128-bits

typedef struct
{
  int                cmd;

  unsigned int       size;
  unsigned int       chunks_per_wb;
  unsigned int       chunks_per_spe;

  unsigned long long ea_input;
  unsigned long long ea_output;
  unsigned int       in_blk_stride;
  unsigned int       out_blk_stride;

  fft_dir_type       direction;
  double             scale;
} Fft_params;



typedef struct
{
  int                cmd;

  unsigned int       size;
  unsigned int       chunks_per_wb;
  unsigned int       chunks_per_spe;

  unsigned long long ea_input_re;
  unsigned long long ea_input_im;
  unsigned long long ea_output_re;
  unsigned long long ea_output_im;
  unsigned int       in_blk_stride;
  unsigned int       out_blk_stride;

  fft_dir_type       direction;
  double             scale;
} Fft_split_params;


#endif // VSIP_OPT_CBE_FFT_PARAMS_H

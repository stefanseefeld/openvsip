/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/pwarp_params.h
    @author  Jules Bergmann
    @date    2007-11-19
    @brief   VSIPL++ Library: Parameters for PWarp kernels.
*/

#ifndef VSIP_OPT_CBE_PWARP_PARAMS_H
#define VSIP_OPT_CBE_PWARP_PARAMS_H

/***********************************************************************
  Definitions
***********************************************************************/

#define VSIP_IMPL_CBE_PWARP_BUFFER_SIZE (256*512)

// Structures used in DMAs should be sized in multiples of 128-bits

typedef struct
{
  float P[9];			// perspective warp matrix
  int   pad[3];

  unsigned long long ea_in;	// input block EA
  unsigned long long ea_out;	// output block EA

  unsigned int in_row_0;	// input origin row
  unsigned int in_col_0;	// input origin column
  unsigned int in_rows;		// input number of rows
  unsigned int in_cols;		// input number of cols
  unsigned int in_stride_0;	// input stride to next row

  unsigned int out_row_0;	// output origin row
  unsigned int out_col_0;	// output origin column
  unsigned int out_rows;	// output number of rows
  unsigned int out_cols;	// output number of cols
  unsigned int out_stride_0;	// output stride to next row
} Pwarp_params;

#endif // VSIP_OPT_CBE_FFT_PARAMS_H

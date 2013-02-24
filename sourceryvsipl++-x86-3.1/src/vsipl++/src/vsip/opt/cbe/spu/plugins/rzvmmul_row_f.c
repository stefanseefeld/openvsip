/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/spu/plugins/rzvmmul_row_f.c
    @author  Don McCoy
    @date    2009-12-17
    @brief   VSIPL++ Library: Vector-Matrix Multiply Kernel for split-
             complex vector, scalar matrix
*/

/***********************************************************************
  Includes
***********************************************************************/

#include <spu_intrinsics.h>
#include <assert.h>
#include <alf_accel.h>
#include <cml.h>

#include <vsip/opt/cbe/vmmul_params.h>
#include <vsip/opt/cbe/dma.h>
#include <lwp_kernel.h>



/***********************************************************************
  Definitions
***********************************************************************/

#define _ALF_MAX_SINGLE_DT_SIZE 16*1024

float save_a_re[VSIP_IMPL_MAX_VMMUL_SIZE] __attribute__((aligned(128)));
float save_a_im[VSIP_IMPL_MAX_VMMUL_SIZE] __attribute__((aligned(128)));



static inline void
add_vector_f(lwp_functions* pf,
	     void*             entries, 
	     alf_data_addr64_t ea,
	     unsigned long     length)
{
  unsigned long const max_length = _ALF_MAX_SINGLE_DT_SIZE / sizeof(float);

  while (length > 0)
  {
    unsigned long cur_length = (length > max_length) ? max_length : length;
    (pf->f_dtl_entry_add)(entries, cur_length, ALF_DATA_FLOAT, ea);
    length -= cur_length;
    ea     += cur_length * sizeof(float);
  } 
}



int
input(lwp_functions* pf,
      void*          p_params, 
      void*          entries, 
      unsigned int   current_count, 
      unsigned int   total_count)
{
  unsigned int const  FP     = 1; // Split-complex data: 1 floats per point.
  Vmmul_split_params* params = (Vmmul_split_params*)p_params;
  unsigned long       length = params->length;
  unsigned long       mult = params->mult;

  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);

  add_vector_f(pf, entries, 
	       params->ea_input_matrix_re +
	       current_count * FP * mult * params->input_stride * sizeof(float),
	       length * mult);

  if (current_count == 0)
  {
    unsigned long dma_size = VSIP_IMPL_INCREASE_TO_DMA_SIZE(length, float);

    add_vector_f(pf, entries, params->ea_input_vector_re, dma_size);
    add_vector_f(pf, entries, params->ea_input_vector_im, dma_size);
  }
  (pf->f_dtl_end)(entries);

  return 0;
}



int
output(lwp_functions* pf,
       void*             p_params, 
       void*             entries, 
       unsigned int      current_count, 
       unsigned int      total_count)
{
  unsigned int const  FP     = 1; // Split-complex data: 1 floats per point.
  Vmmul_split_params* params = (Vmmul_split_params*)p_params;
  unsigned long       length = params->length;
  unsigned long       mult = params->mult;

  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);

  add_vector_f(pf, entries, 
	       params->ea_output_matrix_re +
	       current_count * FP * mult * params->output_stride * sizeof(float),
	       length * mult);
  add_vector_f(pf, entries, 
	       params->ea_output_matrix_im +
	       current_count * FP * mult * params->output_stride * sizeof(float),
	       length * mult);

  (pf->f_dtl_end)(entries);

  return 0;
}






int
kernel(lwp_functions* pf,
       void*             p_params,
       void*             inout,
       unsigned int      iter,
       unsigned int      iter_count)
{
  Vmmul_split_params* params = (Vmmul_split_params*)p_params;
  unsigned long       size   = params->length;
  unsigned long mult = params->mult;

  // If the vector is not long enough for a DMA, it is automatically
  // increased, taking up slightly more buffer space.  We must account
  // for this padding when computing the addresses below.
  unsigned long const vec_size = VSIP_IMPL_INCREASE_TO_DMA_SIZE(size, float);
  assert(VSIP_IMPL_IS_DMA_SIZE(vec_size, float));

  // The matrix is either sent as one long row (mult == 1) or as
  // several rows at a time (mult == N).  The function invoking
  // this kernel ensures these sizes are ok to DMA.
  unsigned long const mat_size = size * mult;
  assert(VSIP_IMPL_IS_DMA_SIZE(mat_size, float));

  assert(mat_size >= VSIP_IMPL_MIN_VMMUL_SIZE);
  assert(mat_size <= VSIP_IMPL_MAX_VMMUL_SIZE);

  // a is vector, b is matrix.  The matrix is sent first.
  float *b_re = (float *)inout + 0;
  float *a_re = (float *)inout + 1 * mat_size;
  float *a_im = (float *)inout + 1 * mat_size + vec_size;
  float *r_re = (float *)inout + 0;
  float *r_im = (float *)inout + 1 * mat_size;

  // The vector is only sent on the first iteration.
  if (iter == 0)
  {
    int i;
    for (i = 0; i < size; ++i)
    {
      save_a_re[i] = a_re[i + params->shift];
      save_a_im[i] = a_im[i + params->shift];
    }
  }

  // Perform the computation.
  int i;
  int m;
  for (m = 0; m < mult; ++m)
    for (i = 0; i < size; ++i)
    {
      r_re[m * size + i] = save_a_re[i] * b_re[m * size + i];
      r_im[m * size + i] = save_a_im[i] * b_re[m * size + i];
    }

  return 0;
}

/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/spu/plugin/zvmmul_row_f.c
    @author  Jules Bergmann
    @date    2008-11-07
    @brief   VSIPL++ Library: Split-complex vector-matrix-multiply kernel.
             2009-02-19 plugin-ified
*/

/***********************************************************************
  Includes
***********************************************************************/

#include <spu_intrinsics.h>
#include <assert.h>
#include <alf_accel.h>
#include <cml.h>

#include <vsip/opt/cbe/vmmul_params.h>
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
      void*             p_params, 
      void*             entries, 
      unsigned int      current_count, 
      unsigned int      total_count)
{
  unsigned int const  FP     = 1; // Split-complex data: 1 floats per point.
  Vmmul_split_params* params = (Vmmul_split_params*)p_params;
  unsigned long       length = params->length;

  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);

  add_vector_f(pf, entries, 
	       params->ea_input_matrix_re +
	       current_count * FP * params->input_stride * sizeof(float),
	       length);
  add_vector_f(pf, entries, 
	       params->ea_input_matrix_im +
	       current_count * FP * params->input_stride * sizeof(float),
	       length);
  if (current_count == 0)
  {
    add_vector_f(pf, entries, params->ea_input_vector_re, length);
    add_vector_f(pf, entries, params->ea_input_vector_im, length);
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

  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);

  add_vector_f(pf, entries, 
	       params->ea_output_matrix_re +
	       current_count * FP * params->output_stride * sizeof(float),
	       length);
  add_vector_f(pf, entries, 
	       params->ea_output_matrix_im +
	       current_count * FP * params->output_stride * sizeof(float),
	       length);

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

  assert(size >= VSIP_IMPL_MIN_VMMUL_SIZE);
  assert(size <= VSIP_IMPL_MAX_VMMUL_SIZE);


  float *a_re = (float *)inout + 2 * size;
  float *a_im = (float *)inout + 3 * size;
  float *b_re = (float *)inout + 0 * size;
  float *b_im = (float *)inout + 1 * size;
  float *r_re = (float *)inout + 0 * size;
  float *r_im = (float *)inout + 1 * size;

  if (iter == 0)
  {
    int i;
    for (i=0; i<size; ++i)
    {
      save_a_re[i] = a_re[i+params->shift];
      save_a_im[i] = a_im[i+params->shift];
    }
  }

  cml_zvmul1_f(save_a_re, save_a_im, b_re, b_im, r_re, r_im, size);

  return 0;
}

/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/spu/plugin/zvmmul_col_f.c
    @author  Jules Bergmann
    @date    2009-03-06
    @brief   VSIPL++ Library: Split-complex vector-matrix-multiply kernel.
*/

/***********************************************************************
  Includes
***********************************************************************/

#include <spu_intrinsics.h>
#include <alf_accel.h>
#include <cml.h>

#include <vsip/opt/cbe/vmmul_params.h>
#include <lwp_kernel.h>



/***********************************************************************
  Definitions
***********************************************************************/

#define _ALF_MAX_SINGLE_DT_SIZE 16*1024
#define VSIP_IMPL_MAX_VMMUL_COL_SIZE 4096
float save_a_re[VSIP_IMPL_MAX_VMMUL_COL_SIZE] __attribute__((aligned(128)));
float save_a_im[VSIP_IMPL_MAX_VMMUL_COL_SIZE] __attribute__((aligned(128)));

inline void
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

  //  Set v_len to minimum of 4 since transfers must be a multiple of 16B.
  unsigned long       v_len  = total_count;
  v_len += params->shift;
  if      (v_len == 0)     v_len = 4;
  else if (v_len % 4 != 0) v_len += 4 - (v_len % 4);

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
    add_vector_f(pf, entries, params->ea_input_vector_re, v_len);
    add_vector_f(pf, entries, params->ea_input_vector_im, v_len);
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


static void
zsvmul1_f(
  float        a_re,
  float        a_im,
  float*       b_re,
  float*       b_im,
  float*       r_re,
  float*       r_im,
  unsigned int length)
{
#if 0 || USE_FUNCTIONAL_VERSION
  int i;
  float tmp;
  for (i=0; i<length; ++i)
  {
    tmp     = a_re * b_re[i] - a_im * b_im[i];
    r_im[i] = a_re * b_im[i] + a_im * b_re[i];
    r_re[i] = tmp;
  }
#else
  typedef vector float vf;
 
  float tmp;
#if 0
  vector float v_a_re = spu_splats(a_re);
  vector float v_a_im = spu_splats(a_im);

  while (length >= 32)
  {
    vf tr0 = v_a_re * *(vf*)&b_re[0]  - v_a_im * *(vf*)&b_im[0];
    vf ti0 = v_a_re * *(vf*)&b_im[0]  + v_a_im * *(vf*)&b_re[0];
    vf tr1 = v_a_re * *(vf*)&b_re[4]  - v_a_im * *(vf*)&b_im[4];
    vf ti1 = v_a_re * *(vf*)&b_im[4]  + v_a_im * *(vf*)&b_re[4];
    vf tr2 = v_a_re * *(vf*)&b_re[8]  - v_a_im * *(vf*)&b_im[8];
    vf ti2 = v_a_re * *(vf*)&b_im[8]  + v_a_im * *(vf*)&b_re[8];
    vf tr3 = v_a_re * *(vf*)&b_re[12] - v_a_im * *(vf*)&b_im[12];
    vf ti3 = v_a_re * *(vf*)&b_im[12] + v_a_im * *(vf*)&b_re[12];
    vf tr4 = v_a_re * *(vf*)&b_re[16] - v_a_im * *(vf*)&b_im[16];
    vf ti4 = v_a_re * *(vf*)&b_im[16] + v_a_im * *(vf*)&b_re[16];
    vf tr5 = v_a_re * *(vf*)&b_re[20] - v_a_im * *(vf*)&b_im[20];
    vf ti5 = v_a_re * *(vf*)&b_im[20] + v_a_im * *(vf*)&b_re[20];
    vf tr6 = v_a_re * *(vf*)&b_re[24] - v_a_im * *(vf*)&b_im[24];
    vf ti6 = v_a_re * *(vf*)&b_im[24] + v_a_im * *(vf*)&b_re[24];
    vf tr7 = v_a_re * *(vf*)&b_re[28] - v_a_im * *(vf*)&b_im[28];
    vf ti7 = v_a_re * *(vf*)&b_im[28] + v_a_im * *(vf*)&b_re[28];

    *(vf*)&r_re[0]  = tr0;
    *(vf*)&r_im[0]  = ti0;
    *(vf*)&r_re[4]  = tr1;
    *(vf*)&r_im[4]  = ti1;
    *(vf*)&r_re[8]  = tr2;
    *(vf*)&r_im[8]  = ti2;
    *(vf*)&r_re[12] = tr3;
    *(vf*)&r_im[12] = ti3;
    *(vf*)&r_re[16] = tr4;
    *(vf*)&r_im[16] = ti4;
    *(vf*)&r_re[20] = tr5;
    *(vf*)&r_im[20] = ti5;
    *(vf*)&r_re[24] = tr6;
    *(vf*)&r_im[24] = ti6;
    *(vf*)&r_re[28] = tr7;
    *(vf*)&r_im[28] = ti7;

    length -= 32;
    b_re += 32; b_im += 32;
    r_re += 32; r_im += 32;
  }
  while (length >= 4)
  {
    vf t0      = v_a_re * *(vf*)b_re - v_a_im * *(vf*)b_im;
    *(vf*)r_im = v_a_re * *(vf*)b_im + v_a_im * *(vf*)b_re;
    *(vf*)r_re = t0;
    length -= 4;
    b_re += 4; b_im += 4;
    r_re += 4; r_im += 4;
  }
#endif
  while (length)
  {
    tmp   = a_re * *b_re - a_im * *b_im;
    *r_im = a_re * *b_im + a_im * *b_re;
    *r_re = tmp;
    --length;
    ++b_re; ++b_im;
    ++r_re; ++r_im;
  }
#endif
}




int
kernel(lwp_functions* pf,
       void*             p_params,
       void*             inout,
       unsigned int      iter,
       unsigned int      iter_count)
{
  Vmmul_split_params* params = (Vmmul_split_params*)p_params;
  int length = params->length;
  unsigned long       v_len  = iter_count;
  v_len += params->shift;
  if      (v_len == 0)     v_len = 4;
  else if (v_len % 4 != 0) v_len += 4 - (v_len % 4);

  float *a_re = (float *)inout + 2*length;
  float *a_im = (float *)inout + 2*length + v_len;
  float *b_re = (float *)inout + 0 * length;
  float *b_im = (float *)inout + 1 * length;
  float *r_re = (float *)inout + 0 * length;
  float *r_im = (float *)inout + 1 * length;

#if 0 || DEBUG
  (pf->f_printf)("zvmmul_col_f (%d/%d): length = %d  v_len = %d\n",
		 iter, iter_count, length, v_len);
#endif


  if (iter == 0)
  {
    int i;
    for (i=0; i<iter_count; ++i)
    {
      save_a_re[i] = a_re[i+params->shift];
      save_a_im[i] = a_im[i+params->shift];
    }
  }

#if DEBUG
  printf("zvmmul_col_f (%d/%d): %f %f\n",
	 iter, iter_count,
	 save_a_re[iter], save_a_im[iter]);
#endif

  zsvmul1_f(save_a_re[iter], save_a_im[iter], b_re, b_im, r_re, r_im, length);

  return 0;
}

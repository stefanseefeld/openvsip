/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/spu/plugin/zfft_f.c
    @author  Jules Bergmann
    @date    2009-02-20
    @brief   VSIPL++ Library: Kernel to compute split-complex float FFT's.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <stdio.h>
#include <alf_accel.h>
#include <assert.h>
#include <spu_mfcio.h>
#include <cml.h>
#include <cml_core.h>

#include <vsip/core/acconfig.hpp>
#include <vsip/opt/cbe/overlay_params.h>
#include <vsip/opt/cbe/fft_params.h>
#include <vsip/opt/cbe/vmmul_params.h>
#include <vsip/opt/cbe/dma.h>
#include <lwp_kernel.h>

#define _ALF_MAX_SINGLE_DT_SIZE 16*1024


/***********************************************************************
  Common Definitions
***********************************************************************/

int current_size = 0;
float buf[2*MAX_FFT_1D_SIZE + MAX_FFT_1D_SIZE + 128/sizeof(float)];



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



/***********************************************************************
  zfft_f Definitions
***********************************************************************/

static int
input_zfft_f(lwp_functions* pf,
	     void*             params,
	     void*             entries,
	     unsigned int      iter,
	     unsigned int      iter_max)
{
  Fft_split_params* fftp   = (Fft_split_params *)params;
  unsigned int      size   = fftp->size;
  unsigned int      chunks = fftp->chunks_per_wb;
  unsigned int      cur_chunks;
  unsigned int      i;
    
  alf_data_addr64_t ea;

  if (iter == iter_max-1 && iter_max * chunks > fftp->chunks_per_spe)
    cur_chunks = fftp->chunks_per_spe % chunks;
  else
    cur_chunks = chunks;
    
  if (size == fftp->in_blk_stride)
  {
    size *= cur_chunks;
    cur_chunks = 1;
  }

  // Transfer input.
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
    
  for (i=0; i<cur_chunks; ++i)
  {
    ea = fftp->ea_input_re +
      (iter * chunks + i) * fftp->in_blk_stride * sizeof(float);
    add_vector_f(pf, entries, ea, size);
  }
    
  for (i=0; i<cur_chunks; ++i)
  {
    ea = fftp->ea_input_im +
      (iter * chunks + i) * fftp->in_blk_stride * sizeof(float);
    add_vector_f(pf, entries, ea, size);
  }
  
  (pf->f_dtl_end)(entries);

  return 0;
}



static int
output_zfft_f(lwp_functions* pf,
	      void*             params,
	      void*             entries,
	      unsigned int      iter,
	      unsigned int      iter_max)
{
  Fft_split_params* fftp   = (Fft_split_params *)params;
  unsigned int      size   = fftp->size;
  unsigned int      chunks = fftp->chunks_per_wb;
  unsigned int      cur_chunks;
  unsigned int      i;
    
  alf_data_addr64_t ea;
    
  if (iter == iter_max-1 && iter_max * chunks > fftp->chunks_per_spe)
    cur_chunks = fftp->chunks_per_spe % chunks;
  else
    cur_chunks = chunks;
  
  if (size == fftp->out_blk_stride)
  {
    size *= cur_chunks;
    cur_chunks = 1;
  }
    
  // Transfer output.
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
    
  for (i=0; i<cur_chunks; ++i)
  {
    ea = fftp->ea_output_re +
      (iter * chunks + i) * fftp->out_blk_stride * sizeof(float);
    add_vector_f(pf, entries, ea, size);
  }
    
  for (i=0; i<cur_chunks; ++i)
  {
    ea = fftp->ea_output_im +
      (iter * chunks + i) * fftp->out_blk_stride * sizeof(float);
    add_vector_f(pf, entries, ea, size);
  }
    
  (pf->f_dtl_end)(entries);

  return 0;
}



static int
kernel_zfft_f(lwp_functions* pf,
	      void*             params,
	      void*             inout,
	      unsigned int      iter,
	      unsigned int      iter_max)
{
  static fft1d_f* obj;

  Fft_split_params* fftp = (Fft_split_params *)params;
  unsigned int      size   = fftp->size;
  unsigned int      chunks = fftp->chunks_per_wb;
  unsigned int      i;
  int dir = fftp->direction == fwd_fft ? CML_FFT_FWD : CML_FFT_INV;

  if (iter == iter_max-1 && iter_max * chunks > fftp->chunks_per_spe)
    chunks = fftp->chunks_per_spe % chunks;

  assert(size >= MIN_FFT_1D_SIZE);
  assert(size <= MAX_FFT_1D_SIZE);

  if (iter == 0 && size != current_size)
  {
#if !NDEBUG
    // Check that buffer space doesn't overlap with stack
    register volatile vector unsigned int get_r1 asm("1");
    unsigned int stack_pointer   = spu_extract(get_r1, 0);
    assert(buf + 2*MAX_FFT_1D_SIZE + size + 128/4 < stack_pointer);
#endif
    int rt = cml_fft1d_setup_f(&obj, CML_FFT_CC, size,
			       buf + 2*MAX_FFT_1D_SIZE);
    assert(rt && obj != NULL);
    current_size = size;
  }

  float* inout_re  = (float*)inout + 0        * size;
  float* inout_im  = (float*)inout + 1*chunks * size;

  for (i=0; i<chunks; ++i)
  {
    cml_zzfft1d_op_f(obj,
		     (float*)inout_re  + i*size, (float*)inout_im  + i*size,
		     (float*)inout_re  + i*size, (float*)inout_im  + i*size,
		     dir, buf);
  }

  if (fftp->scale != (double)1.f)
  {
    // Instead of regular split svmul:
    // cml_core_rzsvmul1_f(fftp->scale, out_re,out_im,out_re,out_im,size);
    // Take advantage of real and imag being contiguous:
    cml_core_svmul1_f(fftp->scale, inout_re, inout_re, 2*size*chunks);
  }

  return 0;
}



/***********************************************************************
  zvmmul_row_f Definitions
***********************************************************************/

static int
input_zvmmul_row_f(lwp_functions* pf,
		   void*             p_params, 
		   void*             entries, 
		   unsigned int      current_count, 
		   unsigned int      total_count)
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
  add_vector_f(pf, entries, 
	       params->ea_input_matrix_im +
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



static int
output_zvmmul_row_f(lwp_functions* pf,
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






static int
kernel_zvmmul_row_f(lwp_functions* pf,
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

  float *b_re = (float *)inout + 0;
  float *b_im = (float *)inout + 1 * mat_size;
  float *a_re = (float *)inout + 2 * mat_size;
  float *a_im = (float *)inout + 2 * mat_size + vec_size;
  float *r_re = (float *)inout + 0;
  float *r_im = (float *)inout + 1 * mat_size;

  float* save_a_re = buf;
  float* save_a_im = buf + vec_size;

  if (iter == 0)
  {
    int i;

    for (i=0; i<size; ++i)
    {
      save_a_re[i] = a_re[i+params->shift];
      save_a_im[i] = a_im[i+params->shift];
    }
  }

  int i;
  for (i = 0; i < mult; ++i)
    cml_zvmul1_f(save_a_re, save_a_im, 
      &b_re[i * size], &b_im[i * size],
      &r_re[i * size], &r_im[i * size], size);

  return 0;
}



int
input(lwp_functions* pf,
      void*             params, 
      void*             entries, 
      unsigned int      current_count, 
      unsigned int      total_count)
{
  switch (((Vmmul_split_params*)params)->cmd)
  {
  case overlay_zvmmul_row_f:
    return input_zvmmul_row_f(pf, params, entries, current_count, total_count);
  case overlay_zfft_f:
    return input_zfft_f(pf, params, entries, current_count, total_count);
  }
  return 1;
}



int
output(lwp_functions* pf,
       void*             params, 
       void*             entries, 
       unsigned int      current_count, 
       unsigned int      total_count)
{
  switch (((Vmmul_split_params*)params)->cmd)
  {
  case overlay_zvmmul_row_f:
    return output_zvmmul_row_f(pf, params, entries, current_count, total_count);
  case overlay_zfft_f:
    return output_zfft_f(pf, params, entries, current_count, total_count);
  }
  return 1;
}



int
kernel(lwp_functions* pf,
       void*             params,
       void*             inout,
       unsigned int      iter,
       unsigned int      iter_count)
{
  switch (((Vmmul_split_params*)params)->cmd)
  {
  case overlay_zvmmul_row_f:
    return kernel_zvmmul_row_f(pf, params, inout, iter, iter_count);
  case overlay_zfft_f:
    return kernel_zfft_f(pf, params, inout, iter, iter_count);
  }
  return 1;
}

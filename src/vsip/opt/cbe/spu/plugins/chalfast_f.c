/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/spu/plugin/chalfast_f.c
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
#include <vsip/opt/cbe/fft_params.h>
#include <vsip/opt/cbe/vmmul_params.h>
#include <vsip/opt/cbe/overlay_params.h>
#include <lwp_kernel.h>

#define _ALF_MAX_SINGLE_DT_SIZE 16*1024


int current_size = 0;
float buf[2*MAX_FFT_1D_SIZE + MAX_FFT_1D_SIZE + 128/sizeof(float)];



/***********************************************************************
  cfft_f Definitions
***********************************************************************/

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

static int
input_cfft_f(lwp_functions* pf,
	     void*             params,
	     void*             entries,
	     unsigned int      iter,
	     unsigned int      iter_max)
{
  Fft_params*  fftp   = (Fft_params *)params;
  unsigned int size   = fftp->size;
  unsigned int chunks = fftp->chunks_per_wb;
  unsigned int cur_chunks;
  unsigned int i;
    
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
    ea = fftp->ea_input +
      (iter * chunks + i) * fftp->in_blk_stride * 2*sizeof(float);
    add_vector_f(pf, entries, ea, 2*size);
  }
  
  (pf->f_dtl_end)(entries);

  return 0;
}



static int
output_cfft_f(lwp_functions* pf,
	      void*             params,
	      void*             entries,
	      unsigned int      iter,
	      unsigned int      iter_max)
{
  Fft_params* fftp   = (Fft_params *)params;
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
    ea = fftp->ea_output +
      (iter * chunks + i) * fftp->out_blk_stride * 2*sizeof(float);
    add_vector_f(pf, entries, ea, 2*size);
  }

  (pf->f_dtl_end)(entries);
    
  return 0;
}



static int
kernel_cfft_f(lwp_functions* pf,
	      void*             params,
	      void*             vinout,
	      unsigned int      iter,
	      unsigned int      iter_max)
{
  static fft1d_f* obj;

  Fft_params*     fftp   = (Fft_params *)params;
  unsigned int    size   = fftp->size;
  unsigned int    chunks = fftp->chunks_per_wb;
  unsigned int    i;
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

  float* inout = (float*)vinout;

  for (i=0; i<chunks; ++i)
  {
    float* t_inout  = inout + i*size*2;

    cml_ccfft1d_op_f(obj, t_inout, t_inout, CML_FFT_FWD, buf);

    if (fftp->direction == fwd_fft)
    {
      if (fftp->scale != (double)1.f)
	cml_core_rcsvmul1_f(fftp->scale, t_inout, t_inout, size);
    }
    else
    {
      // Code for the inverse FFT taken from the CBE SDK Libraries
      // Overview and Users Guide, sec. 8.1.
      int const vec_size = 4;
      vector float* start = (vector float*)t_inout;
      vector float* end   = start + 2 * size / vec_size;
      vector float  s0, s1, e0, e1;
      vector unsigned int mask = (vector unsigned int){-1, -1, 0, 0};
      vector float vscale = spu_splats((float)fftp->scale);
      unsigned int i;

      // Scale the output vector and swap the order of the outputs.
      // Note: there are two float values for each of 'n' complex values.
      s0 = e1 = *start;
      for (i = 0; i < size / vec_size; ++i) 
      {
	s1 = *(start + 1);
	e0 = *(--end);
	
	*start++ = spu_mul(spu_sel(e0, e1, mask), vscale);
	*end     = spu_mul(spu_sel(s0, s1, mask), vscale);
	s0 = s1;
	e1 = e0;
      }
    }
  }

  return 0;
}



/***********************************************************************
  cvmmul_row_f Definitions
***********************************************************************/

static int
input_cvmmul_row_f(lwp_functions* pf,
		   void*             p_params, 
		   void*             entries, 
		   unsigned int      current_count, 
		   unsigned int      total_count)
{
  unsigned int const  wpp    = 2; // Inter-complex data: 1 word per point.
  Vmmul_params*       params = (Vmmul_params*)p_params;
  unsigned long       length = params->length;

  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
  add_vector_f(pf, entries, 
	       params->ea_input_matrix +
	       current_count * params->input_stride * wpp * sizeof(float),
	       length*wpp);
  (pf->f_dtl_end)(entries);

  if (current_count == 0)
  {
    (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, length * wpp * sizeof(float));
    add_vector_f(pf, entries, params->ea_input_vector, length*wpp);
    (pf->f_dtl_end)(entries);
  }


  return 0;
}



static int
output_cvmmul_row_f(lwp_functions* pf,
		    void*             p_params, 
		    void*             entries, 
		    unsigned int      current_count, 
		    unsigned int      total_count)
{
  unsigned int const  wpp    = 2; // Inter-complex data: 1 word per point.
  Vmmul_params*       params = (Vmmul_params*)p_params;
  unsigned long       length = params->length;

  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);

  add_vector_f(pf, entries, 
	       params->ea_output_matrix +
	       current_count * wpp * params->output_stride * sizeof(float),
	       length*wpp);

  (pf->f_dtl_end)(entries);

  return 0;
}






static int
kernel_cvmmul_row_f(lwp_functions* pf,
		    void*             p_params,
		    void*             inout,
		    unsigned int      iter,
		    unsigned int      iter_count)
{
  Vmmul_params* params = (Vmmul_params*)p_params;
  unsigned long       size   = params->length;

  assert(size >= VSIP_IMPL_MIN_VMMUL_SIZE);
  assert(size <= VSIP_IMPL_MAX_VMMUL_SIZE);


  float *a = (float *)inout + 2*size;
  float *b = (float *)inout + 0*size;
  float *r = (float *)inout + 0*size;

  float* save_a = buf;

  if (iter == 0)
  {
    int i;

    for (i=0; i<size; ++i)
    {
      save_a[2*i+0] = a[2*(i+params->shift)+0];
      save_a[2*i+1] = a[2*(i+params->shift)+1];
    }
  }

  cml_cvmul1_f(save_a, b, r, size);

  return 0;
}



/***********************************************************************
  Gateway Definitions
***********************************************************************/

int
input(lwp_functions* pf,
      void*             p_params, 
      void*             entries, 
      unsigned int      current_count, 
      unsigned int      total_count)
{
  switch (((Vmmul_params*)p_params)->cmd)
  {
  case overlay_cvmmul_row_f:
    return input_cvmmul_row_f(pf, p_params, entries,
			      current_count, total_count);
  case overlay_cfft_f:
    return input_cfft_f(pf, p_params, entries,
			current_count, total_count);
  }
  return 1;
}



int
output(lwp_functions* pf,
       void*             p_params, 
       void*             entries, 
       unsigned int      current_count, 
       unsigned int      total_count)
{
  switch (((Vmmul_params*)p_params)->cmd)
  {
  case overlay_cvmmul_row_f:
    return output_cvmmul_row_f(pf, p_params, entries,
			       current_count, total_count);
  case overlay_cfft_f:
    return output_cfft_f(pf, p_params, entries,
			 current_count, total_count);
  }
  return 1;
}



int
kernel(lwp_functions* pf,
       void*             p_params,
       void*             inout,
       unsigned int      iter,
       unsigned int      iter_count)
{
  switch (((Vmmul_params*)p_params)->cmd)
  {
  case overlay_cvmmul_row_f:
    return kernel_cvmmul_row_f(pf, p_params, inout, iter, iter_count);
  case overlay_cfft_f:
    return kernel_cfft_f(pf, p_params, inout, iter, iter_count);
  }
  return 1;
}

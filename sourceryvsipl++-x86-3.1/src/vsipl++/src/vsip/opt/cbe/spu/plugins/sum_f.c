/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <alf_accel.h>

#include <vsip/opt/cbe/reduction_params.h>
#include <lwp_kernel.h>

int input(lwp_functions* pf,
	  void*             params,
	  void*             entries,
	  unsigned int      iter,
	  unsigned int      iter_max)
{
  Reduction_params *p = (Reduction_params*)params;

  alf_data_addr64_t ea;
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
  switch (p->cmd)
  {
    case SUM:
    case SUMSQ:
      // Transfer input A
      ea = p->a_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
    case CSUM:
    case CSUMSQ:
      // Transfer input A
      ea = p->a_ptr + iter * p->length * 2 * sizeof(float);
      (pf->f_dtl_entry_add)(entries, 2 * p->length, ALF_DATA_FLOAT, ea);
      break;
    case ZSUM:
    case ZSUMSQ:
    {
      Reduction_split_params *p = (Reduction_split_params*)params;
      // Transfer input A_re
      ea = p->re_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      // Transfer input A_im
      ea = p->im_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
    }
  }
  (pf->f_dtl_end)(entries);
  return 0;
}

int output(lwp_functions* pf,
	   void*             params,
	   void*             entries,
	   unsigned int      iter,
	   unsigned int      iter_max)
{
  Reduction_params *p = (Reduction_params*)params;
  alf_data_addr64_t ea;
  
  // Transfer output R.
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
  // Even though there is only a single value to return,
  // we return 4 * sizeof(float) bytes to satisfy DMA constraints.
  ea = p->r_ptr + iter * 4 * sizeof(float);
  (pf->f_dtl_entry_add)(entries, 4, ALF_DATA_FLOAT, ea);
  (pf->f_dtl_end)(entries);
  return 0;
}

int kernel(lwp_functions* pf,
	   void*             params,
	   void*             inout,
	   unsigned int      iter,
	   unsigned int      n)
{
  Reduction_params* p = (Reduction_params*)params;
  float *a = (float *)inout;
  float *r = a; // accumulate the result at the start of inout
  unsigned int i;
  switch (p->cmd)
  {
    case SUM:
      for (i = 1; i != p->length; ++i)
	*r += a[i];
      break;
    case SUMSQ:
      *r = a[0]*a[0];
      for (i = 1; i != p->length; ++i)
	*r += a[i]*a[i];
      break;
    case CSUM:
      for (i = 1; i != p->length; ++i)
      {
	r[0] += a[2*i];
	r[1] += a[2*i+1];
      }
      break;
    case CSUMSQ:
    {
      float tmp = a[0]*a[0] - a[1]*a[1];
      r[1] = 2*a[0]*a[1];
      r[0] = tmp;
      for (i = 1; i != p->length; ++i)
      {
	r[0] += a[2*i]*a[2*i] - a[2*i+1]*a[2*i+1];
	r[1] += 2*a[2*i]*a[2*i+1];
      }
      break;
    }
    case ZSUM:
    {
      float *a_re = a;
      float *a_im = a + p->length;
      // be careful here: r aliases (the start of) a_re,
      // so we have to use temporaries for the first two iterations.
      r[0] = a_re[0] + a_re[1];
      r[1] = a_im[0] + a_im[1];
      for (i = 2; i != p->length; ++i)
      {
	r[0] += a_re[i];
	r[1] += a_im[i];
      }
      break;
    }
    case ZSUMSQ:
    {
      float *a_re = a;
      float *a_im = a + p->length;
      // be careful here: r aliases (the start of) a_re,
      // so we have to use temporaries for the first two iterations.
      float r_re = a_re[0]*a_re[0] - a_im[0]*a_im[0];
      float r_im = 2 * a_re[0] * a_im[0];
      r[0] = r_re + a_re[1]*a_re[1] - a_im[1]*a_im[1];
      r[1] = r_im + 2 * a_re[1] * a_im[1];
      for (i = 2; i != p->length; ++i)
      {
	r[0] += a_re[i] * a_re[i] - a_im[i] * a_im[i];
	r[1] += 2 * a_re[i] * a_im[i];
      }
      break;
    }
  }
  return 0;
}

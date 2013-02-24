/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <alf_accel.h>
#include <vsip/opt/cbe/dma.h>
#include <vsip/opt/cbe/hist_params.h>
#include <lwp_kernel.h>

/*
 * We can be more dynamic 'later'.  For now, we silently
 * require this maximum.
 */
static int bin[VSIP_IMPL_HIST_MAX_BINS];

int input(lwp_functions* pf,
	  void*             params,
	  void*             entries,
	  unsigned int      iter,
	  unsigned int      iter_max)
{
  Hist_params *p = (Hist_params*)params;

  /*
   * On the first iteration, initialize the bin vector.
   */
  if (iter == 0)
  {
    int i;
    for (i=0; i < p->num_bin; ++i)
      bin[i] = 0;
  }

  alf_data_addr64_t ea;
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
  switch (p->cmd)
  {
    case HIST:
      // Transfer input value vector
      ea = p->val_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
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
  /*
   * On the last iteration, return the bin vector.
   */
  if (iter == iter_max-1)
  {
    Hist_params *p = (Hist_params*)params;
    alf_data_addr64_t ea;
    
    (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
    // Transfer output bin vector
    ea = p->bin_ptr;
    // The following assumes `sizeof(int) == sizeof(float) == 32 bits`
    (pf->f_dtl_entry_add)(entries,
			  VSIP_IMPL_INCREASE_TO_DMA_SIZE(p->num_bin, float),
			  ALF_DATA_INT32, ea);
    (pf->f_dtl_end)(entries);
  }
  return 0;
}

int kernel(lwp_functions* pf,
	   void*             params,
	   void*             inout,
	   unsigned int      iter,
	   unsigned int      iter_max)
{
  Hist_params* p = (Hist_params*)params;
  int bnum = p->num_bin;
  unsigned int i;
  switch (p->cmd)
  {
    case HIST:
    {
      float *v    = (float *)inout;
      float minv = p->min;
      float maxv = p->max;
      float delta = (maxv - minv) / (bnum - 2);
      for (i=0; i != p->length; ++i)
      {
	float value = v[i];
	int idx = 0;
	if (value >= maxv)
	  idx = bnum - 1;
	else
	  if (value >= minv)
	    idx = (int)(((value - minv) / delta) + 1);
	bin[idx]++;
      }
    }
    break;
  }
  /*
   * On the last iteration, copy to inout the local bin vector.
   */
  if (iter == iter_max-1)
  {
    int* outbin = (int *)inout;
    for (i=0; i < bnum; ++i)
      outbin[i] = bin[i];
  }
  return 0;
}

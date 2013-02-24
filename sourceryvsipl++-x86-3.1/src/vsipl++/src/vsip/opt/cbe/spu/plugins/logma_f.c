/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <spu_intrinsics.h>
#include <alf_accel.h>
#include <vsip/opt/cbe/logma_params.h>
#include <lwp_kernel.h>
#include <simdmath.h>


int input(lwp_functions* pf,
	  void*          params,
	  void*          entries,
	  unsigned int   iter,
	  unsigned int   iter_max)
{
  Logma_params *p = (Logma_params*)params;
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
  switch (p->cmd)
  {
    case LMA:
    {
      alf_data_addr64_t ea;

      // Transfer input A
      ea = p->a_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
    }
  }
  (pf->f_dtl_end)(entries);
  return 0;
}


int output(lwp_functions* pf,
	   void*          params,
	   void*          entries,
	   unsigned int   iter,
	   unsigned int   iter_max)
{
  Logma_params *p = (Logma_params*)params;
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
  switch (p->cmd)
  {
    case LMA:
    {
      alf_data_addr64_t ea;
      
      // Transfer output R.
      ea = p->r_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
    }
  }
  (pf->f_dtl_end)(entries);
  return 0;
}


int kernel(lwp_functions* pf,
	   void*          params,
	   void*          inout,
	   unsigned int   iter,
	   unsigned int   n)
{
  Logma_params* p = (Logma_params*)params;
  switch (p->cmd)
  {
    case LMA:
    {
      int length = p->length / 4;
      vector float *a = (vector float *)inout;
      vector float b = spu_splats((float)p->b_value);
      vector float c = spu_splats((float)p->c_value);
      unsigned int i;

      for (i = 0; i != length; ++i, ++a)
        *a = spu_add(spu_mul(log10f4(*a), b), c);

      return 0;
    }
  }
  return 1;
}

/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#include <alf_accel.h>
#include <spu_intrinsics.h>
#include <simdmath.h>

#include <cml.h>

#include <vsip/opt/cbe/binary_params.h>
#include <lwp_kernel.h>

int input(lwp_functions* pf,
	  void*             params,
	  void*             entries,
	  unsigned int      iter,
	  unsigned int      iter_max)
{
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
      
  switch (((Binary_params*)params)->cmd)
  {
    case ATAN2:
    {
      Binary_params *p = (Binary_params*)params;

      alf_data_addr64_t ea;
      
      // Transfer input A
      ea = p->a_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      
      // Transfer input B.
      ea = p->b_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
    }
    break;

// Placeholder for future binary ops on split-complex.
#if 0
    case XXX:
    {
      Binary_split_params* p = (Binary_split_params*)params;
      alf_data_addr64_t ea;

      // Transfer input A real
      ea = p->a_re_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      
      ea = p->a_im_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      
      // Transfer input B.
      ea = p->b_re_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      
      ea = p->b_im_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
    }
    break;
#endif
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
  alf_data_addr64_t ea;

  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);

  switch (((Binary_params*)params)->cmd)
  {
    case ATAN2:
    {
      Binary_params* p = (Binary_params*)params;
      
      // Transfer output R.
      ea = p->r_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
    }
    break;

// Placeholder for future binary ops on split-complex.
#if 0
    case XXXX:
    {
      Binary_split_params *p = (Binary_split_params*)params;
      // Transfer output R.

      ea = p->r_re_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      
      ea = p->r_im_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
    }
    break;
#endif
  }

  (pf->f_dtl_end)(entries);

  return 0;
}



int kernel(lwp_functions* pf,
	   void*             params,
	   void*             inout,
	   unsigned int      iter,
	   unsigned int      n)
{
  Binary_params* p = (Binary_params*)params;
  int length = p->length;

  switch (p->cmd)
  {
    case ATAN2:
    {
      float* a = (float *)inout + 0 * length;
      float* b = (float *)inout + 1 * length;
      float* r = (float *)inout + 0 * length;
      
      vector float *va = (vector float *)a;
      vector float *vb = (vector float *)b;
      vector float *vr = (vector float *)r;

      unsigned int i;

      length /= 4;

      for (i = 0; i != length; ++i, ++vr, ++va, ++vb)
        *vr = atan2f4(*va,*vb);
      break;
    }
    break;

  }

  return 0;
}

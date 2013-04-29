/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <alf_accel.h>
#include <cml.h>

#include <vsip/opt/cbe/binary_params.h>
#include <lwp_kernel.h>
#include <vsip/opt/cbe/overlay_params.h>

int input(lwp_functions* pf,
	  void*             params,
	  void*             entries,
	  unsigned int      iter,
	  unsigned int      iter_max)
{
  switch (((Binary_params*)params)->cmd)
  {
    case overlay_vmul_f:
    case overlay_cvmul_f:
    {
      Binary_params *p = (Binary_params*)params;

      // words-per-point
      int wpp = p->cmd == overlay_vmul_f ? 1 : 2;

      alf_data_addr64_t ea;
      (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);

      // Transfer input A
      ea = p->a_ptr + iter * p->length * wpp * sizeof(float);
      (pf->f_dtl_entry_add)(entries, wpp * p->length, ALF_DATA_FLOAT, ea);
    
      // Transfer input B.
      ea = p->b_ptr + iter * p->length * wpp * sizeof(float);
      (pf->f_dtl_entry_add)(entries, wpp * p->length, ALF_DATA_FLOAT, ea);

      (pf->f_dtl_end)(entries);
    }
    break;

    case overlay_zvmul_f:
    {
      Binary_split_params* p = (Binary_split_params*)params;
      alf_data_addr64_t ea;

      (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
    
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
    
      (pf->f_dtl_end)(entries);
    }
    break;
  }

  return 0;
}



int output(lwp_functions* pf,
	   void*             params,
	   void*             entries,
	   unsigned int      iter,
	   unsigned int      iter_max)
{
  switch (((Binary_params*)params)->cmd)
  {
    case overlay_vmul_f:
    case overlay_cvmul_f:
    {
      Binary_params *p = (Binary_params*)params;
      int wpp = p->cmd == overlay_vmul_f ? 1 : 2;
      alf_data_addr64_t ea;
      
      // Transfer output R.
      (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
      ea = p->r_ptr + iter *  p->length * wpp * sizeof(float);
      (pf->f_dtl_entry_add)(entries, wpp * p->length, ALF_DATA_FLOAT, ea);
      (pf->f_dtl_end)(entries);
    }
    break;

    case overlay_zvmul_f:
    {
      Binary_split_params* p = (Binary_split_params*)params;
      alf_data_addr64_t ea;

      // Transfer output R.
      (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
    
      ea = p->r_re_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
    
      ea = p->r_im_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
    
      (pf->f_dtl_end)(entries);
    }
    break;
  }

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
    case overlay_vmul_f:
    {
      float *a = (float *)inout + 0 * length;
      float *b = (float *)inout + 1 * length;
      float *r = (float *)inout + 0 * length;
      
      cml_vmul1_f(a, b, r, length);
    }
    break;

    case overlay_cvmul_f:
    {
      float *a = (float *)inout + 0 * length;
      float *b = (float *)inout + 2 * length;
      float *r = (float *)inout + 0 * length;

      cml_cvmul1_f(a, b, r, length);
    }
    break;

    case overlay_zvmul_f:
    {
      float *a_re = (float *)inout + 0 * length;
      float *a_im = (float *)inout + 1 * length;
      float *b_re = (float *)inout + 2 * length;
      float *b_im = (float *)inout + 3 * length;
      float *r_re = (float *)inout + 0 * length;
      float *r_im = (float *)inout + 1 * length;

      cml_zvmul1_f(a_re, a_im, b_re, b_im, r_re, r_im, length);
    }
    break;
  }

  return 0;
}



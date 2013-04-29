/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <alf_accel.h>
#include <cml.h>

#include <vsip/opt/cbe/dot_params.h>
#include <lwp_kernel.h>

int input(lwp_functions* pf,
	  void*             params,
	  void*             entries,
	  unsigned int      iter,
	  unsigned int      iter_max)
{
  Dot_split_params *p = (Dot_split_params*)params;

  alf_data_addr64_t ea;
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);

  // Transfer input A
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
  return 0;
}

int output(lwp_functions* pf,
	   void*             params,
	   void*             entries,
	   unsigned int      iter,
	   unsigned int      iter_max)
{
  Dot_split_params *p = (Dot_split_params*)params;
  alf_data_addr64_t ea;
  
  // Transfer output R.
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
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
  Dot_split_params* p = (Dot_split_params*)params;
  float *a_re = (float *)inout + 0 * p->length;
  float *a_im = (float *)inout + 1 * p->length;
  float *b_re = (float *)inout + 2 * p->length;
  float *b_im = (float *)inout + 3 * p->length;
  float *r = (float *)inout + 0 * p->length;
  float tmp[2];
  if (p->conj)
    cml_zvdotj_f(a_re, a_im, 1, 
		 b_re, b_im, 1,
		 &tmp[0], &tmp[1], p->length);
  else
    cml_zvdot_f(a_re, a_im, 1, 
		b_re, b_im, 1,
		&tmp[0], &tmp[1], p->length);
  r[0] = tmp[0];
  r[1] = tmp[1];
  return 0;
}



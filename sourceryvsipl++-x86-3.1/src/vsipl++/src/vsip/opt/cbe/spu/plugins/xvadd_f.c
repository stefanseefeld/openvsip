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
    case overlay_vadd_f:
    case overlay_cvadd_f:
    {
      Binary_params *p = (Binary_params*)params;

      // words per point
      int wpp = p->cmd == overlay_vadd_f ? 1 : 2;
      alf_data_addr64_t ea;
      
      (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
      
      // Transfer input A
      ea = p->a_ptr + iter * wpp * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, wpp * p->length, ALF_DATA_FLOAT, ea);
      
      // Transfer input B.
      ea = p->b_ptr + iter * wpp * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, wpp * p->length, ALF_DATA_FLOAT, ea);
      
      (pf->f_dtl_end)(entries);
    }
    break;

    case overlay_zvadd_f:
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
  alf_data_addr64_t ea;

  switch (((Binary_params*)params)->cmd)
  {
    case overlay_vadd_f:
    case overlay_cvadd_f:
    {
      Binary_params* p = (Binary_params*)params;
      int wpp = p->cmd == overlay_vadd_f ? 1 : 2;
      alf_data_addr64_t ea;
      
      // Transfer output R.
      (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
      ea = p->r_ptr + iter * wpp * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, wpp * p->length, ALF_DATA_FLOAT, ea);
      (pf->f_dtl_end)(entries);
    }
    break;

    case overlay_zvadd_f:
    {
      Binary_split_params *p = (Binary_split_params*)params;
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


#define uDecl \
  vector float* va = (vector float*)a; \
  vector float* vb = (vector float*)b; \
  vector float* vr = (vector float*)r

#define uLoad(n) vr[n] = va[n] + vb[n]
#define uIncr(n) va += (n); vb += (n); vr += (n);
#define uBump(n) a += (n); b += (n); r += (n);

#define u_4 uLoad(0); uIncr(1); uBump(4)
#define u_8 uLoad(0); uLoad(1); uIncr(2); uBump(8)
#define u_16 uLoad(0); uLoad(1); uLoad(2); uLoad(3); uIncr(4); uBump(16)

#define unRoll(n) uDecl; \
  while (length >= n) \
  { \
    u_##n; \
    length -= n; \
  }

static void
vadd1_f(
  float const* a,
  float const* b,
  float*       r,
  int          length)
{
  unRoll(4)

  while (length > 0)
  {
    *r = *a + *b;
    --length;
    ++r; ++a; ++b;
  }
}

static void
zvadd1_f(
  float const* a_re,
  float const* a_im,
  float const* b_re,
  float const* b_im,
  float*       r_re,
  float*       r_im,
  int          length)
{
  while (length >= 4)
  {
    vector float tr0 = *(vector float*)a_re + *(vector float*)b_re;
    vector float ti0 = *(vector float*)a_im + *(vector float*)b_im;
    *(vector float*)r_re = tr0;
    *(vector float*)r_im = ti0;
    length -= 4;
    r_re += 4; r_im += 4;
    a_re += 4; a_im += 4;
    b_re += 4; b_im += 4;
  }

  while (length > 0)
  {
    *r_re = *a_re + *b_re;
    *r_im = *a_im + *b_im;
    --length;
    ++r_re; ++r_im;
    ++a_re; ++a_im;
    ++b_re; ++b_im;
 }
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
    case overlay_vadd_f:
    {
      float* a = (float *)inout + 0 * length;
      float* b = (float *)inout + 1 * length;
      float* r = (float *)inout + 0 * length;
      
      vadd1_f(a, b, r, length);
    }
    break;

    case overlay_cvadd_f:
    {
      float *a = (float *)inout + 0 * length;
      float *b = (float *)inout + 2 * length;
      float *r = (float *)inout + 0 * length;

      vadd1_f(a, b, r, 2*length);
    }
    break;

    case overlay_zvadd_f:
    {
      float *a_re = (float *)inout + 0 * length;
      float *a_im = (float *)inout + 1 * length;
      float *b_re = (float *)inout + 2 * length;
      float *b_im = (float *)inout + 3 * length;
      float *r_re = (float *)inout + 0 * length;
      float *r_im = (float *)inout + 1 * length;

      zvadd1_f(a_re, a_im, b_re, b_im, r_re, r_im, length);
    }
    break;
  }

  return 0;
}

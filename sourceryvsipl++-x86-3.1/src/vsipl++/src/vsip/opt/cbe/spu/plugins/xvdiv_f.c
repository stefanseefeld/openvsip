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

#include <simdmath.h>
#include <simdmath/divf4.h>
#include <simdmath/divf4_fast.h>

int input(lwp_functions* pf,
	  void*             params,
	  void*             entries,
	  unsigned int      iter,
	  unsigned int      iter_max)
{
  alf_data_addr64_t ea;
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
  switch (((Binary_params*)params)->cmd)
  {
    case overlay_vdiv_f:
    {
      Binary_params *p = (Binary_params*)params;
      // words-per-point
      int wpp = p->cmd == overlay_vdiv_f ? 1 : 2;

      // Transfer input A
      ea = p->a_ptr + iter * p->length * wpp * sizeof(float);
      (pf->f_dtl_entry_add)(entries, wpp * p->length, ALF_DATA_FLOAT, ea);
    
      // Transfer input B.
      ea = p->b_ptr + iter * p->length * wpp * sizeof(float);
      (pf->f_dtl_entry_add)(entries, wpp * p->length, ALF_DATA_FLOAT, ea);
    }
    break;

    case overlay_zvdiv_f:
    {
      Binary_split_params *p = (Binary_split_params*)params;

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
    }
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
  alf_data_addr64_t ea;
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
  switch (((Binary_params*)params)->cmd)
  {
    case overlay_vdiv_f:
    {
      Binary_params *p = (Binary_params*)params;
      int wpp = p->cmd == overlay_vdiv_f ? 1 : 2;
      
      // Transfer output R.
      ea = p->r_ptr + iter *  p->length * wpp * sizeof(float);
      (pf->f_dtl_entry_add)(entries, wpp * p->length, ALF_DATA_FLOAT, ea);
      (pf->f_dtl_end)(entries);
    }
    break;
    case overlay_zvdiv_f:
    {
      Binary_split_params *p = (Binary_split_params*)params;
      
      // Transfer output R.
      ea = p->r_re_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      ea = p->r_im_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
    }
    break;
  }
  (pf->f_dtl_end)(entries);
  return 0;
}

#define uDecl \
  vector float* va = (vector float*)a; \
  vector float* vb = (vector float*)b; \
  vector float* vr = (vector float*)r

// #define uLoad(n) vr[n] = va[n] / vb[n]
#define uLoad(n) vr[n] = _divf4(va[n], vb[n])
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
vdiv1_f(
  float const* a,
  float const* b,
  float*       r,
  int          length)
{
  unRoll(16)

  while (length > 0)
  {
    *r = *a / *b;
    --length;
    ++r; ++a; ++b;
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
    case overlay_vdiv_f:
    {
      float *a = (float *)inout + 0 * length;
      float *b = (float *)inout + 1 * length;
      float *r = (float *)inout + 0 * length;    
      vdiv1_f(a, b, r, length);
      break;
    }
    case overlay_zvdiv_f:
    {   
      float *va_re = (float *)inout;
      float *va_im = va_re + p->length;
      float *vb_re = va_im + p->length;
      float *vb_im = vb_re + p->length;
      vector float *var = (vector float *)va_re;
      vector float *vai = (vector float *)va_im;
      vector float *vbr = (vector float *)vb_re;
      vector float *vbi = (vector float *)vb_im;
      unsigned int i;
      length /= 4;
      for (i = 0; i != length; ++i, ++var, ++vai, ++vbr, ++vbi)
      {   
        vector float den = spu_madd(*vbr,*vbr,spu_mul(*vbi,*vbi)); // br*br+bi*bi
        vector float nur = spu_madd(*var,*vbr,spu_mul(*vai,*vbi)); // ar*br+ai*bi
        vector float nui = spu_msub(*vai,*vbr,spu_mul(*var,*vbi)); // ai*br-ar*bi
	*var = _divf4(nur,den);
	*vai = _divf4(nui,den);
      }   
      break;
    }
  }

  return 0;
}



/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <spu_intrinsics.h>
#include <alf_accel.h>
#include <vsip/opt/cbe/ternary_params.h>
#include <lwp_kernel.h>

int input(lwp_functions* pf,
	  void*             params,
	  void*             entries,
	  unsigned int      iter,
	  unsigned int      iter_max)
{
  Ternary_params *p = (Ternary_params*)params;
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
  switch (p->cmd)
  {
    case AM:
    case MA:
    {
      alf_data_addr64_t ea;

      // Transfer input A
      ea = p->a_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      // Transfer input B
      ea = p->b_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      // Transfer input C
      ea = p->c_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
    }
    case CAM:
    case CMA:
    {
      alf_data_addr64_t ea;

      // Transfer input A
      ea = p->a_ptr + iter * p->length * 2 * sizeof(float);
      (pf->f_dtl_entry_add)(entries, 2 * p->length, ALF_DATA_FLOAT, ea);
      // Transfer input B
      ea = p->b_ptr + iter * p->length * 2 * sizeof(float);
      (pf->f_dtl_entry_add)(entries, 2 * p->length, ALF_DATA_FLOAT, ea);
      // Transfer input C
      ea = p->c_ptr + iter * p->length * 2 * sizeof(float);
      (pf->f_dtl_entry_add)(entries, 2 * p->length, ALF_DATA_FLOAT, ea);
      break;
    }
    case ZAM:
    case ZMA:
    {
      Ternary_split_params *p = (Ternary_split_params*)params;
      alf_data_addr64_t ea;

      // Transfer input A
      ea = p->a_re_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      ea = p->a_im_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      // Transfer input B
      ea = p->b_re_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      ea = p->b_im_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      // Transfer input C
      ea = p->c_re_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      ea = p->c_im_ptr + iter * p->length * sizeof(float);
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
  Ternary_params *p = (Ternary_params*)params;
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
  switch (p->cmd)
  {
    case AM:
    case MA:
    {
      alf_data_addr64_t ea;
      
      // Transfer output R.
      ea = p->r_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
    }
    case CAM:
    case CMA:
    {
      alf_data_addr64_t ea;
      
      // Transfer output R.
      ea = p->r_ptr + iter * p->length * 2 * sizeof(float);
      (pf->f_dtl_entry_add)(entries, 2 * p->length, ALF_DATA_FLOAT, ea);
      break;
    }
    case ZAM:
    case ZMA:
    {
      Ternary_split_params *p = (Ternary_split_params*)params;
      alf_data_addr64_t ea;
      
      // Transfer output R.
      ea = p->r_re_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      ea = p->r_im_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
    }
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
  Ternary_params* p = (Ternary_params*)params;
  switch (p->cmd)
  {
    case AM:
    {
      int length = p->length / 4;
      vector float *a = (vector float *)inout;
      vector float *b = a + length;
      vector float *c = a + 2 * length;
      unsigned int i;
      for (i = 0; i != length; ++i, ++a, ++b, ++c)
	*a = spu_mul(spu_add(*a, *b), *c);
      return 0;
    }
    case MA:
    {
      int length = p->length / 4;
      vector float *a = (vector float *)inout;
      vector float *b = a + length;
      vector float *c = a + 2 * length;
      unsigned int i;
      for (i = 0; i != length; ++i, ++a, ++b, ++c)
	*a = spu_madd(*a, *b, *c);
      return 0;
    }
    case CAM:
    {
      static vector unsigned char lo = 
	(vector unsigned char) { 0, 1, 2, 3, 16, 17, 18, 19,
				 4, 5, 6, 7, 20, 21, 22, 23};

      static vector unsigned char hi = 
	(vector unsigned char) { 8,  9, 10, 11, 24, 25, 26, 27,
				12, 13, 14, 15, 28, 29, 30, 31};

      int length = p->length / 4;
      float *a = (float *)inout;
      float *b = a + 8 * length;
      float *c = a + 16 * length;
      unsigned int i;
      // (a + b) * c:
      // r.r = (a.r+b.r)*c.r - (a.i+b.i)*c.i
      // r.i = (a.r+b.r)*c.i + (a.i+b.i)*c.r
      for (i = 0; i != length; ++i, a+=8, b+=8, c+=8)
      {
	vector float av = {*a, *(a+2), *(a+4), *(a+6)};              // a.r
	vector float bv = {*b, *(b+2), *(b+4), *(b+6)};              // b.r
	vector float cv = {*c, *(c+2), *(c+4), *(c+6)};              // c.r
	vector float dv = {*(a+1), *(a+3), *(a+5), *(a+7)};          // a.i
	vector float ev = {*(b+1), *(b+3), *(b+5), *(b+7)};          // b.i
	vector float fv = {*(c+1), *(c+3), *(c+5), *(c+7)};          // c.i
	vector float trv = spu_add(av, bv); // a.r+b.r
	vector float tiv = spu_add(dv, ev); // a.i+b.i
	vector float sv = spu_mul(trv, cv); // (a.r+b.r)*c.r
	vector float tv = spu_mul(trv, fv); // (a.r+b.r)*c.i
	vector float real = spu_nmsub(tiv, fv, sv); // r.r
	vector float imag = spu_madd(tiv, cv, tv);  // r.i
	// interleave result
	*(vector float *)a = spu_shuffle(real, imag, lo);
	*(vector float *)(a+4) = spu_shuffle(real, imag, hi);
      }
      return 0;
    }
    case CMA:
    {
      static vector unsigned char lo = 
	(vector unsigned char) { 0, 1, 2, 3, 16, 17, 18, 19,
				 4, 5, 6, 7, 20, 21, 22, 23};

      static vector unsigned char hi = 
	(vector unsigned char) { 8,  9, 10, 11, 24, 25, 26, 27,
				12, 13, 14, 15, 28, 29, 30, 31};

      int length = p->length / 4;
      float *a = (float *)inout;
      float *b = a + 8 * length;
      float *c = a + 16 * length;
      unsigned int i;
      // a * b + c:
      // r.r = a.r*b.r + c.r - a.i*b.i
      // r.i = a.r*b.i + c.i + a.i*b.r
      for (i = 0; i != length; ++i, a+=8, b+=8, c+=8)
      {
	vector float av = {*a, *(a+2), *(a+4), *(a+6)};              // a.r
	vector float bv = {*b, *(b+2), *(b+4), *(b+6)};              // b.r
	vector float cv = {*c, *(c+2), *(c+4), *(c+6)};              // c.r
	vector float dv = {*(a+1), *(a+3), *(a+5), *(a+7)};          // a.i
	vector float ev = {*(b+1), *(b+3), *(b+5), *(b+7)};          // b.i
	vector float fv = {*(c+1), *(c+3), *(c+5), *(c+7)};          // c.i
	vector float real = spu_nmsub(dv, ev, spu_madd(av, bv, cv)); // r.r
	vector float imag = spu_madd(dv, bv, spu_madd(av, ev, fv));  // r.i
	// interleave result
	*(vector float *)a = spu_shuffle(real, imag, lo);
	*(vector float *)(a+4) = spu_shuffle(real, imag, hi);
      }
      return 0;
    }
    case ZAM:
    {
      int length = p->length / 4;
      float *a_re = (float *)inout;
      float *a_im = a_re + 4 * length;
      float *b_re = a_re + 8 * length;
      float *b_im = a_re + 12 * length;
      float *c_re = a_re + 16 * length;
      float *c_im = a_re + 20 * length;
      unsigned int i;
      // (a + b) * c:
      // r.r = (a.r+b.r)*c.r - (a.i+b.i)*c.i
      // r.i = (a.r+b.r)*c.i + (a.i+b.i)*c.r
      for (i = 0; i != length;
	   ++i, a_re+=4, b_re+=4, c_re+=4, a_im+=4, b_im+=4, c_im+=4)
      {
	vector float *av = (vector float *)a_re;
	vector float *bv = (vector float *)b_re;
	vector float *cv = (vector float *)c_re;
	vector float *dv = (vector float *)a_im;
	vector float *ev = (vector float *)b_im;
	vector float *fv = (vector float *)c_im;
	vector float trv = spu_add(*av, *bv); // a.r+b.r
	vector float tiv = spu_add(*dv, *ev); // a.i+b.i
	vector float sv = spu_mul(trv, *cv); // (a.r+b.r)*c.r
	vector float tv = spu_mul(trv, *fv); // (a.r+b.r)*c.i
	*av = spu_nmsub(tiv, *fv, sv); // r.r
        *dv = spu_madd(tiv, *cv, tv);  // r.i
      }
      return 0;
    }
    case ZMA:
    {
      int length = p->length / 4;
      float *a_re = (float *)inout;
      float *a_im = a_re + 4 * length;
      float *b_re = a_re + 8 * length;
      float *b_im = a_re + 12 * length;
      float *c_re = a_re + 16 * length;
      float *c_im = a_re + 20 * length;
      unsigned int i;
      // a * b + c:
      // r.r = a.r*b.r + c.r - a.i*b.i
      // r.i = a.r*b.i + c.i + a.i*b.r
      for (i = 0; i != length;
	   ++i, a_re+=4, b_re+=4, c_re+=4, a_im+=4, b_im+=4, c_im+=4)
      {
	vector float *av = (vector float *)a_re;
	vector float *bv = (vector float *)b_re;
	vector float *cv = (vector float *)c_re;
	vector float *dv = (vector float *)a_im;
	vector float *ev = (vector float *)b_im;
	vector float *fv = (vector float *)c_im;
	vector float tmp = spu_nmsub(*dv, *ev, spu_madd(*av, *bv, *cv));
	*dv = spu_madd(*dv, *bv, spu_madd(*av, *ev, *fv));
	*av = tmp;
      }
      return 0;
    }
  }
  return 1;
}

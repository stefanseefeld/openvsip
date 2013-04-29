/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <alf_accel.h>
#include <spu_intrinsics.h>
#include <simdmath.h>
#include <vsip/opt/cbe/spu/vsip.h>
#include <vsip/opt/cbe/unary_params.h>
#include <lwp_kernel.h>

#define NO_VECTORIZATION_YET 1

int input(lwp_functions* pf,
	  void*             params,
	  void*             entries,
	  unsigned int      iter,
	  unsigned int      iter_max)
{
  Unary_params *p = (Unary_params*)params;
  alf_data_addr64_t ea;

  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_IN, 0);
  switch (p->cmd)
  {
    case SQRT:
    case MAG:
    case MAGSQ:
    case MINUS:
    case SQ:
    case ATAN:
    case COS:
    case SIN:
    case LOG:
    case LOG10:
      // Transfer input A
      ea = p->a_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
    case CMAG:
    case CMAGSQ:
    case CSQ:
    case CCOS:
    case CSIN:
    case CLOG:
    case CLOG10:
    case CCONJ:
    case CMINUS:
      // Transfer input A
      ea = p->a_ptr + iter * p->length * 2 * sizeof(float);
      (pf->f_dtl_entry_add)(entries, 2 * p->length, ALF_DATA_FLOAT, ea);
      break;
    case ZMAG:
    case ZMAGSQ:
    case ZSQ:
    case ZCONJ:
    case ZMINUS:
    case ZCOS:
    case ZSIN:
    case ZLOG:
    case ZLOG10:
    {
      Unary_split_params *p = (Unary_split_params*)params;
      // Transfer input A
      ea = p->a_re_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      ea = p->a_im_ptr + iter * p->length * sizeof(float);
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
  Unary_params *p = (Unary_params*)params;
  alf_data_addr64_t ea;
  (pf->f_dtl_begin)(entries, ALF_BUF_OVL_OUT, 0);
  switch (p->cmd)
  {
    case SQRT:
    case MAG:
    case MAGSQ:
    case CMAG:
    case CMAGSQ:
    case MINUS:
    case SQ:
    case ATAN:
    case COS:
    case SIN:
    case LOG:
    case LOG10:
      ea = p->r_ptr + iter *  p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
    case CSQ:
    case CCOS:
    case CSIN:
    case CLOG:
    case CLOG10:
    case CCONJ:
    case CMINUS:
      ea = p->r_ptr + iter * p->length * 2 * sizeof(float);
      (pf->f_dtl_entry_add)(entries, 2 * p->length, ALF_DATA_FLOAT, ea);
      break;
    case ZMAG:
    case ZMAGSQ:
    {
      // mag and magsq only return a float array.
      Unary_split_params *p = (Unary_split_params*)params;
      ea = p->r_re_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      break;
    }
    case ZSQ:
    case ZCONJ:
    case ZMINUS:
    case ZCOS:
    case ZSIN:
    case ZLOG:
    case ZLOG10:
    {
      Unary_split_params *p = (Unary_split_params*)params;
      ea = p->r_re_ptr + iter * p->length * sizeof(float);
      (pf->f_dtl_entry_add)(entries, p->length, ALF_DATA_FLOAT, ea);
      ea = p->r_im_ptr + iter * p->length * sizeof(float);
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
  Unary_params* p = (Unary_params*)params;
  int length = p->length / 4;

  vector float *v = (vector float *)inout;
  unsigned int i;

  switch (p->cmd)
  {
    case SQRT:
      for (i = 0; i != length; ++i, ++v)
	*v = sqrtf4(*v);
      break;
    case MAG:
      for (i = 0; i != length; ++i, ++v)
	*v = fabsf4(*v);
      break;
    case MINUS:
      for (i = 0; i != length; ++i, ++v)
	*v = negatef4(*v);
      break;
    case MAGSQ:
    case SQ:
      for (i = 0; i != length; ++i, ++v)
	*v = spu_mul(*v,*v);
      break;
    case ATAN:
      for (i = 0; i != length; ++i, ++v)
	*v = atanf4(*v);
      break;
    case COS:
      for (i = 0; i != length; ++i, ++v)
	*v = cosf4(*v);
      break;
    case SIN:
      for (i = 0; i != length; ++i, ++v)
	*v = sinf4(*v);
      break;
    case LOG:
      for (i = 0; i != length; ++i, ++v)
	*v = logf4(*v);
      break;
    case LOG10:
      for (i = 0; i != length; ++i, ++v)
	*v = log10f4(*v);
      break;
    case CMAG:
    {
#if NO_VECTORIZATION_YET
      float *v = (float *)inout;
      for (i = 0; i != p->length; ++i)
	v[i] = sqrt(v[2*i]*v[2*i] + v[2*i+1]*v[2*i+1]);
#else
#endif
      break;
    }
    case CMAGSQ:
    {
#if NO_VECTORIZATION_YET
      float *v = (float *)inout;
      for (i = 0; i != p->length; ++i)
	v[i] = v[2*i]*v[2*i] + v[2*i+1]*v[2*i+1];
#else
#endif
      break;
    }
    case CSQ:
    {
#if NO_VECTORIZATION_YET
      float *v = (float *)inout;
      for (i = 0; i != p->length; ++i)
      {
	float tmp = v[2*i]*v[2*i] - v[2*i+1]*v[2*i+1];
	v[2*i+1] = 2*v[2*i]*v[2*i+1];
	v[2*i] = tmp;
      }
#else
#endif
      break;
    }
    case CCOS:
      cvcos(inout, p->length);
      break;
    case CSIN:
      cvsin(inout, p->length);
      break;
    case CLOG:
      cvlog(inout, p->length);
      break;
    case CLOG10:
      cvlog10(inout, p->length);
      break;
    case ZMAG:
    {
      float *v_re = (float *)inout;
      float *v_im = v_re + p->length;
      vector float *vr = (vector float *)v_re;
      vector float *vi = (vector float *)v_im;
      for (i = 0; i != length; ++i, ++vr, ++vi)
      {
	*vr = sqrtf4(spu_madd(*vr,*vr,spu_mul(*vi,*vi)));
      }
      break;
    }
    case ZMAGSQ:
    {
      float *v_re = (float *)inout;
      float *v_im = v_re + p->length;
      vector float *vr = (vector float *)v_re;
      vector float *vi = (vector float *)v_im;
      for (i = 0; i != length; ++i, ++vr, ++vi)
      {
	*vr = spu_madd(*vr,*vr,spu_mul(*vi,*vi));
      }
      break;
    }
    case ZSQ:
    {
      float *v_re = (float *)inout;
      float *v_im = v_re + p->length;
      vector float *vr = (vector float *)v_re;
      vector float *vi = (vector float *)v_im;
      for (i = 0; i != length; ++i, ++vr, ++vi)
      {
	vector float tmp = spu_mul(*vr,*vi);
	*vr = spu_nmsub(*vi,*vi,spu_mul(*vr,*vr));	// vr = vr*vr - vi*vi
	*vi = spu_add(tmp,tmp);				// vi = vr*vi + vr*vi
      }
      break;
    }
    case ZCOS:
      zvcos(inout, p->length);
      break;
    case ZSIN:
      zvsin(inout, p->length);
      break;
    case ZLOG:
      zvlog(inout, p->length);
      break;
    case ZLOG10:
      zvlog10(inout, p->length);
      break;
    case CCONJ:
    {
#if NO_VECTORIZATION_YET
      float *iv = (float *)inout;
      for (i = 0; i != p->length; ++i)
	iv[2*i+1] = -iv[2*i+1];
#else
#endif
      break;
    }
    case CMINUS:
    {
#if NO_VECTORIZATION_YET
      float *iv = (float *)inout;
      for (i = 0; i != p->length; ++i)
      {
	iv[2*i] = -iv[2*i];
	iv[2*i+1] = -iv[2*i+1];
      }
#else
#endif
      break;
    }
    case ZCONJ:
    {
      float *a_im = (float *)inout + 4*length;
      vector float *ari = (vector float *)a_im;
      for (i = 0; i != length; ++i, ++ari)
      {
	*ari = negatef4(*ari);
      }
      break;
    }
    case ZMINUS:
    {
      float *a_re = (float *)inout + 0*length;
      float *a_im = a_re + 4*length;
      vector float *vr = (vector float *)a_re;
      vector float *vi = (vector float *)a_im;
      for (i = 0; i != length; ++i, ++vr, ++vi)
      {
	*vr = negatef4(*vr);
	*vi = negatef4(*vi);
      }
      break;
    }
  }
  return 0;
}



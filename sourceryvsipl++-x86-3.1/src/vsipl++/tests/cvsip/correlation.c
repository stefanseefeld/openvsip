/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    correlation.c
    @author  Jules Bergmann, Stefan Seefeld
    @date    2008-06-16
    @brief   VSIPL++ Library: Unit tests for [signal.correl] items.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip.h>
#include "test.h"
#include "output.h"
#include <assert.h>

#define VERBOSE 1

/***********************************************************************
  Definitions
***********************************************************************/

vsip_length
ref_corr_output_size(vsip_support_region supp, vsip_length M, vsip_length N)
{
  if (supp == VSIP_SUPPORT_FULL) return N + M - 1;
  else if (supp == VSIP_SUPPORT_SAME) return N;
  else return N - M + 1;
}

vsip_stride
ref_expected_shift(vsip_support_region supp, vsip_length M)
{
  if (supp == VSIP_SUPPORT_FULL) return -(M-1);
  else if (supp == VSIP_SUPPORT_SAME) return -(M/2);
  else return 0;
}

void
ref_corr_d(vsip_bias bias, vsip_support_region sup,
           vsip_vview_d const *ref,
           vsip_vview_d const *in,
           vsip_vview_d const *out)
{
  vsip_length M = vsip_vgetlength_d(ref);
  vsip_length N = vsip_vgetlength_d(in);
  vsip_length P = vsip_vgetlength_d(out);
  vsip_length expected_P = ref_corr_output_size(sup, M, N);
  vsip_stride shift      = ref_expected_shift(sup, M);

  assert(expected_P == P);

  vsip_vview_d *sub = vsip_vcreate_d(M, VSIP_MEM_NONE);

  // compute correlation
  vsip_index i;
  for (i=0; i<P; ++i)
  {
    vsip_vfill_d(0, sub);
    vsip_stride pos = (vsip_stride)i + shift;
    double scale;
    if (pos < 0)
    {
      vsip_vview_d *subsub = vsip_vsubview_d(sub, -pos, M + pos);
      vsip_vview_d *insub = vsip_vsubview_d(in, 0, M + pos);
      vsip_vcopy_d_d(insub, subsub);
      vsip_vdestroy_d(subsub);
      vsip_vdestroy_d(insub);
      scale = M + pos;
    }
    else if (pos + M > N)
    {
      vsip_vview_d *subsub = vsip_vsubview_d(sub, 0, N - pos);
      vsip_vview_d *insub = vsip_vsubview_d(in, pos, N - pos);
      vsip_vcopy_d_d(insub, subsub);
      vsip_vdestroy_d(subsub);
      vsip_vdestroy_d(insub);
      scale = N - pos;
    }
    else
    {
      vsip_vview_d *insub = vsip_vsubview_d(in, pos, M);
      vsip_vcopy_d_d(insub, sub);
      vsip_vdestroy_d(insub);
      scale = M;
    }

#if VSIP_IMPL_CORR_CORRECT_SAME_SUPPORT_SCALING
#else
    if (sup == VSIP_SUPPORT_SAME)
    {
      if      (i < (M/2))     scale = i + (M+1)/2;         // i + ceil(M/2)
      else if (i < N - (M/2)) scale = M;                   // M
      else                    scale = N - 1 + (M+1)/2 - i; // N-1+ceil(M/2)-i
    }
#endif
      
    double val = vsip_vdot_d(ref, sub);
    if (bias == VSIP_UNBIASED) val /= scale;
    vsip_vput_d(out, i, val);
  }
}

void
ref_ccorr_d(vsip_bias bias, vsip_support_region sup,
            vsip_cvview_d const *ref,
            vsip_cvview_d const *in,
            vsip_cvview_d const *out)
{
  vsip_length M = vsip_cvgetlength_d(ref);
  vsip_length N = vsip_cvgetlength_d(in);
  vsip_length P = vsip_cvgetlength_d(out);
  vsip_length expected_P = ref_corr_output_size(sup, M, N);
  vsip_stride shift      = ref_expected_shift(sup, M);

  assert(expected_P == P);

  vsip_cvview_d *sub = vsip_cvcreate_d(M, VSIP_MEM_NONE);

  // compute correlation
  vsip_index i;
  for (i=0; i<P; ++i)
  {
    vsip_cvfill_d(vsip_cmplx_d(0,0), sub);
    vsip_stride pos = (vsip_stride)i + shift;
    double scale;
    if (pos < 0)
    {
      vsip_cvview_d *subsub = vsip_cvsubview_d(sub, -pos, M + pos);
      vsip_cvview_d *insub = vsip_cvsubview_d(in, 0, M + pos);
      vsip_cvcopy_d_d(insub, subsub);
      vsip_cvdestroy_d(subsub);
      vsip_cvdestroy_d(insub);
      scale = M + pos;
    }
    else if (pos + M > N)
    {
      vsip_cvview_d *subsub = vsip_cvsubview_d(sub, 0, N - pos);
      vsip_cvview_d *insub = vsip_cvsubview_d(in, pos, N - pos);
      vsip_cvcopy_d_d(insub, subsub);
      vsip_cvdestroy_d(subsub);
      vsip_cvdestroy_d(insub);
      scale = N - pos;
    }
    else
    {
      vsip_cvview_d *insub = vsip_cvsubview_d(in, pos, M);
      vsip_cvcopy_d_d(insub, sub);
      vsip_cvdestroy_d(insub);
      scale = M;
    }

#if VSIP_IMPL_CORR_CORRECT_SAME_SUPPORT_SCALING
#else
    if (sup == VSIP_SUPPORT_SAME)
    {
      if      (i < (M/2))     scale = i + (M+1)/2;         // i + ceil(M/2)
      else if (i < N - (M/2)) scale = M;                   // M
      else                    scale = N - 1 + (M+1)/2 - i; // N-1+ceil(M/2)-i
    }
#endif
      
    vsip_cscalar_d val = vsip_cvjdot_d(ref, sub);
    if (bias == VSIP_UNBIASED)
    {
      val.r /= scale;
      val.i /= scale;
    }
    vsip_cvput_d(out, i, val);
  }
}


/// Test general 1-D correlation.

void
test_corr_d(vsip_support_region support, vsip_bias bias,
            vsip_length ref_size, vsip_length input_size)
{
  vsip_length const n_loop = 3;
  vsip_length const output_size = ref_corr_output_size(support, ref_size, input_size);

  vsip_corr1d_d *corr = vsip_corr1d_create_d(ref_size, input_size, support, 0, VSIP_ALG_SPACE);
  vsip_corr1d_attr attr;
  vsip_corr1d_getattr_d(corr, &attr);

  test_assert(attr.support  == support);
  test_assert(attr.ref_len == ref_size);
  test_assert(attr.data_len == input_size);
  test_assert(attr.lag_len == output_size);

  vsip_randstate *rand = vsip_randcreate(0, 1, 1, VSIP_PRNG);

  vsip_vview_d *ref = vsip_vcreate_d(ref_size, VSIP_MEM_NONE);
  vsip_vview_d *in = vsip_vcreate_d(input_size, VSIP_MEM_NONE);
  vsip_vview_d *out = vsip_vcreate_d(output_size, VSIP_MEM_NONE);
  vsip_vfill_d(100, out);
  vsip_vview_d *chk = vsip_vcreate_d(output_size, VSIP_MEM_NONE);
  vsip_vfill_d(101, chk);

  vsip_index loop;
  for (loop=0; loop<n_loop; ++loop)
  {
    if (loop == 0)
    {
      vsip_vfill_d(1, ref);
      vsip_vramp_d(0, 1, in);
    }
    else if (loop == 1)
    {
      vsip_vrandu_d(rand, ref);
      vsip_vramp_d(0, 1, in);
    }
    else
    {
      vsip_vrandu_d(rand, ref);
      vsip_vrandu_d(rand, in);
    }

    vsip_correlate1d_d(corr, bias, ref, in, out);

    ref_corr_d(bias, support, ref, in, chk);

    double error = verror_db_d(out, chk);

#if VERBOSE
    if (error > -100)
    {
      vsip_index i;
      for (i=0; i<output_size; ++i)
      {
        printf("%d : out = %f, chk = %f\n", i, vsip_vget_d(out, i), vsip_vget_d(chk, i));
      }
      printf("error = %f\n", error);
    }
#endif

    test_assert(error < -100);
  }
}

void
test_ccorr_d(vsip_support_region support, vsip_bias bias,
             vsip_length ref_size, vsip_length input_size)
{
  vsip_length const n_loop = 3;
  vsip_length const output_size = ref_corr_output_size(support, ref_size, input_size);

  vsip_ccorr1d_d *corr = vsip_ccorr1d_create_d(ref_size, input_size, support, 0, VSIP_ALG_SPACE);
  vsip_ccorr1d_attr attr;
  vsip_ccorr1d_getattr_d(corr, &attr);

  test_assert(attr.support  == support);
  test_assert(attr.ref_len == ref_size);
  test_assert(attr.data_len == input_size);
  test_assert(attr.lag_len == output_size);

  vsip_randstate *rand = vsip_randcreate(0, 1, 1, VSIP_PRNG);

  vsip_cvview_d *ref = vsip_cvcreate_d(ref_size, VSIP_MEM_NONE);
  vsip_cvview_d *in = vsip_cvcreate_d(input_size, VSIP_MEM_NONE);
  vsip_cvview_d *out = vsip_cvcreate_d(output_size, VSIP_MEM_NONE);
  vsip_cvfill_d(vsip_cmplx_d(100,0), out);
  vsip_cvview_d *chk = vsip_cvcreate_d(output_size, VSIP_MEM_NONE);
  vsip_cvfill_d(vsip_cmplx_d(101,0), chk);

  vsip_index loop;
  for (loop=0; loop<n_loop; ++loop)
  {
    if (loop == 0)
    {
      vsip_cvfill_d(vsip_cmplx_d(1,0), ref);
      vsip_cvramp_d(vsip_cmplx_d(0,0), vsip_cmplx_d(1,0), in);
    }
    else if (loop == 1)
    {
      vsip_cvrandu_d(rand, ref);
      vsip_cvramp_d(vsip_cmplx_d(0,0), vsip_cmplx_d(1,0), in);
    }
    else
    {
      vsip_cvrandu_d(rand, ref);
      vsip_cvrandu_d(rand, in);
    }

    vsip_ccorrelate1d_d(corr, bias, ref, in, out);

    ref_ccorr_d(bias, support, ref, in, chk);

    double error = cverror_db_d(out, chk);

#if VERBOSE
    if (error > -100)
    {
      vsip_index i;
      for (i=0; i<output_size; ++i)
      {
        vsip_cscalar_d out_value = vsip_cvget_d(out, i);
        vsip_cscalar_d chk_value = vsip_cvget_d(chk, i);
        printf("%d : out = (%f, %f), chk = (%f, %f)\n", i, out_value.r, out_value.i, chk_value.r, chk_value.i);
      }
      printf("error = %f\n", error);
    }
#endif

    test_assert(error < -100);
  }
}



/// Test general 1-D correlation.

void
corr_cases_d(vsip_length M, vsip_length N)
{
  test_corr_d(VSIP_SUPPORT_MIN, VSIP_BIASED, M, N);
  test_corr_d(VSIP_SUPPORT_MIN, VSIP_UNBIASED, M, N);

  test_corr_d(VSIP_SUPPORT_SAME, VSIP_BIASED, M, N);
  test_corr_d(VSIP_SUPPORT_SAME, VSIP_UNBIASED, M, N);

  test_corr_d(VSIP_SUPPORT_FULL, VSIP_BIASED, M, N);
  test_corr_d(VSIP_SUPPORT_FULL, VSIP_UNBIASED, M, N);
}

void
ccorr_cases_d(vsip_length M, vsip_length N)
{
  test_ccorr_d(VSIP_SUPPORT_MIN, VSIP_BIASED, M, N);
  test_ccorr_d(VSIP_SUPPORT_MIN, VSIP_UNBIASED, M, N);

  test_ccorr_d(VSIP_SUPPORT_SAME, VSIP_BIASED, M, N);
  test_ccorr_d(VSIP_SUPPORT_SAME, VSIP_UNBIASED, M, N);

  test_ccorr_d(VSIP_SUPPORT_FULL, VSIP_BIASED, M, N);
  test_ccorr_d(VSIP_SUPPORT_FULL, VSIP_UNBIASED, M, N);
}


void
corr_cover_d()
{
  corr_cases_d(8, 8);

  corr_cases_d(1, 128);
  corr_cases_d(7, 128);
  corr_cases_d(8, 128);
  corr_cases_d(9, 128);

  corr_cases_d(7, 127);
  corr_cases_d(8, 127);
  corr_cases_d(9, 127);

  corr_cases_d(7, 129);
  corr_cases_d(8, 129);
  corr_cases_d(9, 129);
}

void
ccorr_cover_d()
{
  ccorr_cases_d(8, 8);

  ccorr_cases_d(1, 128);
  ccorr_cases_d(7, 128);
  ccorr_cases_d(8, 128);
  ccorr_cases_d(9, 128);

  ccorr_cases_d(7, 127);
  ccorr_cases_d(8, 127);
  ccorr_cases_d(9, 127);

  ccorr_cases_d(7, 129);
  ccorr_cases_d(8, 129);
  ccorr_cases_d(9, 129);
}

int
main(int argc, char** argv)
{
  vsip_init(0);
  corr_cover_d();
  ccorr_cover_d();
  return 0;
}

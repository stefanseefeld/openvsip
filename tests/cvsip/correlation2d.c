//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is made available under the GPL.
// See the accompanying file LICENSE.GPL for details.

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
           vsip_mview_d *ref,
           vsip_mview_d *in,
           vsip_mview_d *out)
{
  vsip_length Mr = vsip_mgetcollength_d(ref);
  vsip_length Mc = vsip_mgetrowlength_d(ref);
  vsip_length Nr = vsip_mgetcollength_d(in);
  vsip_length Nc = vsip_mgetrowlength_d(in);
  vsip_length Pr = vsip_mgetcollength_d(out);
  vsip_length Pc = vsip_mgetrowlength_d(out);

  vsip_length expected_Pr = ref_corr_output_size(sup, Mr, Nr);
  vsip_length expected_Pc = ref_corr_output_size(sup, Mc, Nc);
  vsip_stride shift_r     = ref_expected_shift(sup, Mr);
  vsip_stride shift_c     = ref_expected_shift(sup, Mc);

  assert(expected_Pr == Pr);
  assert(expected_Pc == Pc);

  vsip_mview_d *sub = vsip_mcreate_d(Mr, Mc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mview_d *tmp = vsip_mcreate_d(Mr, Mc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_stride sub_r_index;
  vsip_length sub_r_length;
  vsip_stride sub_c_index;
  vsip_length sub_c_length;
  vsip_stride in_r_index;
  vsip_length in_r_length;
  vsip_stride in_c_index;
  vsip_length in_c_length;

  // compute correlation
  vsip_index r, c;
  for (r=0; r<Pr; ++r)
  {
    vsip_stride pos_r = (vsip_stride)r + shift_r;

    for (c=0; c<Pc; ++c)
    {
      vsip_stride pos_c = (vsip_stride)c + shift_c;
      double scale = 1;

      if (pos_r < 0)
      {
	sub_r_index = -pos_r;
        sub_r_length = Mr + pos_r;
	in_r_index = 0;
        in_r_length = Mr+pos_r;
	scale *= Mr + pos_r;
      }
      else if (pos_r + Mr > Nr)
      {
	sub_r_index = 0;
        sub_r_length = Nr-pos_r;
	in_r_index = pos_r;
        in_r_length = Nr-pos_r;
	scale *= Nr - pos_r;
      }
      else
      {
	sub_r_index = 0;
        sub_r_length = Mr;
	in_r_index = pos_r;
        in_r_length = Mr;
	scale *= Mr;
      }

      if (pos_c < 0)
      {
	sub_c_index = -pos_c;
        sub_c_length = Mc + pos_c;
	in_c_index = 0;
        in_c_length = Mc+pos_c;
	scale *= Mc + pos_c;
      }
      else if (pos_c + Mc > Nc)
      {
	sub_c_index = 0;
        sub_c_length = Nc-pos_c;
	in_c_index = pos_c;
        in_c_length = Nc-pos_c;
	scale *= Nc - pos_c;
      }
      else
      {
	sub_c_index = 0;
        sub_c_length = Mc;
	in_c_index = pos_c;
        in_c_length = Mc;
	scale *= Mc;
      }

      vsip_mfill_d(0, sub);

      vsip_mview_d *subsub = vsip_msubview_d(sub, sub_r_index, sub_c_index, sub_r_length, sub_c_length);
      vsip_mview_d *insub = vsip_msubview_d(in, in_r_index, in_c_index, in_r_length, in_c_length);
      vsip_mcopy_d_d(insub, subsub);
      vsip_mdestroy_d(subsub);
      vsip_mdestroy_d(insub);
      
      vsip_mmul_d(ref, sub, tmp);
      double val = vsip_msumval_d(tmp);
      if (bias == VSIP_UNBIASED) val /= scale;
      
      vsip_mput_d(out, r, c, val);
    }
  }
  vsip_malldestroy_d(tmp);
  vsip_malldestroy_d(sub);
}

void
ref_ccorr_d(vsip_bias bias, vsip_support_region sup,
            vsip_cmview_d *ref,
            vsip_cmview_d *in,
            vsip_cmview_d *out)
{
  vsip_length Mr = vsip_cmgetcollength_d(ref);
  vsip_length Mc = vsip_cmgetrowlength_d(ref);
  vsip_length Nr = vsip_cmgetcollength_d(in);
  vsip_length Nc = vsip_cmgetrowlength_d(in);
  vsip_length Pr = vsip_cmgetcollength_d(out);
  vsip_length Pc = vsip_cmgetrowlength_d(out);

  vsip_length expected_Pr = ref_corr_output_size(sup, Mr, Nr);
  vsip_length expected_Pc = ref_corr_output_size(sup, Mc, Nc);
  vsip_stride shift_r     = ref_expected_shift(sup, Mr);
  vsip_stride shift_c     = ref_expected_shift(sup, Mc);

  assert(expected_Pr == Pr);
  assert(expected_Pc == Pc);

  vsip_cmview_d *sub = vsip_cmcreate_d(Mr, Mc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmview_d *tmp = vsip_cmcreate_d(Mr, Mc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_stride sub_r_index;
  vsip_length sub_r_length;
  vsip_stride sub_c_index;
  vsip_length sub_c_length;
  vsip_stride in_r_index;
  vsip_length in_r_length;
  vsip_stride in_c_index;
  vsip_length in_c_length;

  // compute correlation
  vsip_index r, c;
  for (r=0; r<Pr; ++r)
  {
    vsip_stride pos_r = (vsip_stride)r + shift_r;

    for (c=0; c<Pc; ++c)
    {
      vsip_stride pos_c = (vsip_stride)c + shift_c;
      double scale = 1;

      if (pos_r < 0)
      {
	sub_r_index = -pos_r;
        sub_r_length = Mr + pos_r;
	in_r_index = 0;
        in_r_length = Mr+pos_r;
	scale *= Mr + pos_r;
      }
      else if (pos_r + Mr > Nr)
      {
	sub_r_index = 0;
        sub_r_length = Nr-pos_r;
	in_r_index = pos_r;
        in_r_length = Nr-pos_r;
	scale *= Nr - pos_r;
      }
      else
      {
	sub_r_index = 0;
        sub_r_length = Mr;
	in_r_index = pos_r;
        in_r_length = Mr;
	scale *= Mr;
      }

      if (pos_c < 0)
      {
	sub_c_index = -pos_c;
        sub_c_length = Mc + pos_c;
	in_c_index = 0;
        in_c_length = Mc+pos_c;
	scale *= Mc + pos_c;
      }
      else if (pos_c + Mc > Nc)
      {
	sub_c_index = 0;
        sub_c_length = Nc-pos_c;
	in_c_index = pos_c;
        in_c_length = Nc-pos_c;
	scale *= Nc - pos_c;
      }
      else
      {
	sub_c_index = 0;
        sub_c_length = Mc;
	in_c_index = pos_c;
        in_c_length = Mc;
	scale *= Mc;
      }

      vsip_cmfill_d(vsip_cmplx_d(0, 0), sub);

      vsip_cmview_d *subsub = vsip_cmsubview_d(sub, sub_r_index, sub_c_index, sub_r_length, sub_c_length);
      vsip_cmview_d *insub = vsip_cmsubview_d(in, in_r_index, in_c_index, in_r_length, in_c_length);
      vsip_cmcopy_d_d(insub, subsub);
      vsip_cmdestroy_d(subsub);
      vsip_cmdestroy_d(insub);
      
      vsip_cmjmul_d(ref, sub, tmp);
      vsip_cscalar_d val = vsip_cmsumval_d(tmp);
      if (bias == VSIP_UNBIASED)
      {
        val.r /= scale;
        val.i /= scale;
      }
      vsip_cmput_d(out, r, c, val);
    }
  }
  vsip_cmalldestroy_d(tmp);
  vsip_cmalldestroy_d(sub);
}

void
test_corr_d(vsip_support_region support, vsip_bias bias,
            vsip_length ref_rows, vsip_length ref_cols,
            vsip_length in_rows, vsip_length in_cols)
{
  vsip_length const n_loop = 3;

  vsip_length const Pr = ref_corr_output_size(support, ref_rows, in_rows);
  vsip_length const Pc = ref_corr_output_size(support, ref_cols, in_cols);

  vsip_corr2d_d *corr = vsip_corr2d_create_d(ref_rows, ref_cols, in_rows, in_cols,
                                             support, 0, VSIP_ALG_SPACE);
  vsip_corr2d_attr attr;
  vsip_corr2d_getattr_d(corr, &attr);

  test_assert(attr.support  == support);
  test_assert(attr.ref_len.r == ref_rows);
  test_assert(attr.ref_len.c == ref_cols);
  test_assert(attr.data_len.r == in_rows);
  test_assert(attr.data_len.c == in_cols);
  test_assert(attr.lag_len.r == Pr);
  test_assert(attr.lag_len.c == Pc);

  vsip_randstate *rand = vsip_randcreate(0, 1, 1, VSIP_PRNG);

  vsip_mview_d *ref = vsip_mcreate_d(ref_rows, ref_cols, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mview_d *in = vsip_mcreate_d(in_rows, in_cols, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mview_d *out = vsip_mcreate_d(Pr, Pc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mfill_d(100, out);
  vsip_mview_d *chk = vsip_mcreate_d(Pr, Pc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mfill_d(101, chk);

  vsip_index loop;
  for (loop=0; loop<n_loop; ++loop)
  {
    if (loop == 0)
    {
      vsip_mfill_d(1, ref);
      vsip_index r;
      for (r=0; r<in_rows; ++r)
      {
        vsip_vview_d *row = vsip_mrowview_d(in, r);
	vsip_vramp_d(0, 1, row);
        vsip_vdestroy_d(row);
      }
    }
    else if (loop == 1)
    {
      vsip_mrandu_d(rand, ref);
      vsip_index r;
      for (r=0; r<in_rows; ++r)
      {
        vsip_vview_d *row = vsip_mrowview_d(in, r);
	vsip_vramp_d(0, 1, row);
        vsip_vdestroy_d(row);
      }
    }
    else
    {
      vsip_mrandu_d(rand, ref);
      vsip_mrandu_d(rand, in);
    }

    vsip_correlate2d_d(corr, bias, ref, in, out);

    ref_corr_d(bias, support, ref, in, chk);

    double error = merror_db_d(out, chk);

#if VERBOSE
    if (error > -100)
    {
      vsip_index r;
      vsip_index c;
      for (r=0; r<Pr; ++r)
        for (c=0; c<Pc; ++c)
        {
          printf("%d,%d : out = %f, chk = %f\n", r, c, vsip_mget_d(out, r, c), vsip_mget_d(chk, r, c));
        }
      printf("error = %f\n", error);
    }
#endif
    test_assert(error < -100);
  }
}

void
test_ccorr_d(vsip_support_region support, vsip_bias bias,
             vsip_length ref_rows, vsip_length ref_cols,
             vsip_length in_rows, vsip_length in_cols)
{
  vsip_length const n_loop = 3;

  vsip_length const Pr = ref_corr_output_size(support, ref_rows, in_rows);
  vsip_length const Pc = ref_corr_output_size(support, ref_cols, in_cols);

  vsip_ccorr2d_d *corr = vsip_ccorr2d_create_d(ref_rows, ref_cols, in_rows, in_cols,
                                               support, 0, VSIP_ALG_SPACE);
  vsip_ccorr2d_attr attr;
  vsip_ccorr2d_getattr_d(corr, &attr);

  test_assert(attr.support  == support);
  test_assert(attr.ref_len.r == ref_rows);
  test_assert(attr.ref_len.c == ref_cols);
  test_assert(attr.data_len.r == in_rows);
  test_assert(attr.data_len.c == in_cols);
  test_assert(attr.lag_len.r == Pr);
  test_assert(attr.lag_len.c == Pc);

  vsip_randstate *rand = vsip_randcreate(0, 1, 1, VSIP_PRNG);

  vsip_cmview_d *ref = vsip_cmcreate_d(ref_rows, ref_cols, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmview_d *in = vsip_cmcreate_d(in_rows, in_cols, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmview_d *out = vsip_cmcreate_d(Pr, Pc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmfill_d(vsip_cmplx_d(100, 0), out);
  vsip_cmview_d *chk = vsip_cmcreate_d(Pr, Pc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmfill_d(vsip_cmplx_d(101, 0), chk);

  vsip_index loop;
  for (loop=0; loop<n_loop; ++loop)
  {
    if (loop == 0)
    {
      vsip_cmfill_d(vsip_cmplx_d(1, 0), ref);
      vsip_index r;
      for (r=0; r<in_rows; ++r)
      {
        vsip_cvview_d *row = vsip_cmrowview_d(in, r);
	vsip_cvramp_d(vsip_cmplx_d(0, 0), vsip_cmplx_d(1, 0), row);
        vsip_cvdestroy_d(row);
      }
    }
    else if (loop == 1)
    {
      vsip_cmrandu_d(rand, ref);
      vsip_index r;
      for (r=0; r<in_rows; ++r)
      {
        vsip_cvview_d *row = vsip_cmrowview_d(in, r);
	vsip_cvramp_d(vsip_cmplx_d(0, 0), vsip_cmplx_d(1, 0), row);
        vsip_cvdestroy_d(row);
      }
    }
    else
    {
      vsip_cmrandu_d(rand, ref);
      vsip_cmrandu_d(rand, in);
    }

    vsip_ccorrelate2d_d(corr, bias, ref, in, out);

    ref_ccorr_d(bias, support, ref, in, chk);

    double error = cmerror_db_d(out, chk);

#if VERBOSE
    if (error > -100)
    {
      vsip_index r;
      vsip_index c;
      for (r=0; r<Pr; ++r)
        for (c=0; c<Pc; ++c)
        {
          vsip_cscalar_d out_value = vsip_cmget_d(out, r, c);
          vsip_cscalar_d chk_value = vsip_cmget_d(chk, r, c);
          printf("%d,%d : out = (%f, %f), chk = (%f, %f)\n", r, c, out_value.r, out_value.i, chk_value.r, chk_value.i);
        }
      printf("error = %f\n", error);
    }
#endif
    test_assert(error < -100);
  }
}

void
corr_cases_d(vsip_length ref_rows, vsip_length ref_cols,
             vsip_length in_rows, vsip_length in_cols)
{
  test_corr_d(VSIP_SUPPORT_MIN, VSIP_BIASED, ref_rows, ref_cols, in_rows, in_cols);
  test_corr_d(VSIP_SUPPORT_MIN, VSIP_UNBIASED, ref_rows, ref_cols, in_rows, in_cols);

  test_corr_d(VSIP_SUPPORT_SAME, VSIP_BIASED, ref_rows, ref_cols, in_rows, in_cols);
  test_corr_d(VSIP_SUPPORT_SAME, VSIP_UNBIASED, ref_rows, ref_cols, in_rows, in_cols);

  test_corr_d(VSIP_SUPPORT_FULL, VSIP_BIASED, ref_rows, ref_cols, in_rows, in_cols);
  test_corr_d(VSIP_SUPPORT_FULL, VSIP_UNBIASED, ref_rows, ref_cols, in_rows, in_cols);
}

void
ccorr_cases_d(vsip_length ref_rows, vsip_length ref_cols,
              vsip_length in_rows, vsip_length in_cols)
{
  test_ccorr_d(VSIP_SUPPORT_MIN, VSIP_BIASED, ref_rows, ref_cols, in_rows, in_cols);
  test_ccorr_d(VSIP_SUPPORT_MIN, VSIP_UNBIASED, ref_rows, ref_cols, in_rows, in_cols);

  test_ccorr_d(VSIP_SUPPORT_SAME, VSIP_BIASED, ref_rows, ref_cols, in_rows, in_cols);
  test_ccorr_d(VSIP_SUPPORT_SAME, VSIP_UNBIASED, ref_rows, ref_cols, in_rows, in_cols);

  test_ccorr_d(VSIP_SUPPORT_FULL, VSIP_BIASED, ref_rows, ref_cols, in_rows, in_cols);
  test_ccorr_d(VSIP_SUPPORT_FULL, VSIP_UNBIASED, ref_rows, ref_cols, in_rows, in_cols);
}

void
corr_cover_d()
{
  corr_cases_d(8, 8, 8, 8);

  corr_cases_d(1, 1, 32, 32);
  corr_cases_d(2, 4, 32, 32);
  corr_cases_d(2, 3, 32, 32);
  corr_cases_d(3, 2, 32, 32);

  corr_cases_d(1, 1, 16, 13);
  corr_cases_d(2, 4, 16, 13);
  corr_cases_d(2, 3, 16, 13);
  corr_cases_d(3, 2, 16, 13);

  corr_cases_d(1, 1, 13, 16);
  corr_cases_d(2, 4, 13, 16);
  corr_cases_d(2, 3, 13, 16);
  corr_cases_d(3, 2, 13, 16);
}

void
ccorr_cover_d()
{
  ccorr_cases_d(8, 8, 8, 8);

  ccorr_cases_d(1, 1, 32, 32);
  ccorr_cases_d(2, 4, 32, 32);
  ccorr_cases_d(2, 3, 32, 32);
  ccorr_cases_d(3, 2, 32, 32);

  ccorr_cases_d(1, 1, 16, 13);
  ccorr_cases_d(2, 4, 16, 13);
  ccorr_cases_d(2, 3, 16, 13);
  ccorr_cases_d(3, 2, 16, 13);

  ccorr_cases_d(1, 1, 13, 16);
  ccorr_cases_d(2, 4, 13, 16);
  ccorr_cases_d(2, 3, 13, 16);
  ccorr_cases_d(3, 2, 13, 16);
}

int
main(int argc, char** argv)
{
  vsip_init(0);
  corr_cover_d();
  ccorr_cover_d();
  return 0;
}

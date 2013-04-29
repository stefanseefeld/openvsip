/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    convolution2d.c
    @author  Jules Bergmann, Stefan Seefeld
    @date    2008-06-16
    @brief   VSIPL++ Library: Unit tests for [signal.convolution] items.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip.h>
#include "test.h"
#include "output.h"
#include <assert.h>

#define VERBOSE 1

// The following is defined in vsip/core/signal/conv_common.hpp,
// and not visible here, so we have to redefine it for consistency.
#define VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE 1
double const ERROR_THRESH = -70;

vsip_length expected_output_size(vsip_support_region supp,
                                 vsip_length M,    // kernel length
                                 vsip_length N,    // input  length
                                 vsip_length D)    // decimation factor
{
  if (supp == VSIP_SUPPORT_FULL) return ((N + M - 2)/D) + 1;
  else if (supp == VSIP_SUPPORT_SAME) return ((N - 1)/D) + 1;
  else
  {
#if VSIP_IMPL_CONV_CORRECT_MIN_SUPPORT_SIZE
    return ((N - M + 1) / D) + ((N - M + 1) % D == 0 ? 0 : 1);
#else
    return ((N - 1)/D) - ((M-1)/D) + 1;
#endif
  }
}

vsip_length expected_shift(vsip_support_region supp,
                           vsip_length M, vsip_length D)
{
  if (supp == VSIP_SUPPORT_FULL) return 0;
  else if (supp == VSIP_SUPPORT_SAME) return (M/2);
  else return (M-1);
}


void
init_in_d(vsip_mview_d *in, int k)
{
  vsip_length rows = vsip_mgetcollength_d(in);
  vsip_length cols = vsip_mgetrowlength_d(in);
  vsip_index i, j;
  for (i=0; i<rows; ++i)
    for (j=0; j<cols; ++j)
      vsip_mput_d(in, i, j, i*rows+j+k);
}

void
cinit_in_d(vsip_cmview_d *in, int k)
{
  vsip_length rows = vsip_cmgetcollength_d(in);
  vsip_length cols = vsip_cmgetrowlength_d(in);
  vsip_index i, j;
  for (i=0; i<rows; ++i)
    for (j=0; j<cols; ++j)
      vsip_cmput_d(in, i, j, vsip_cmplx_d(i*rows+j+k, 0));
}

/// Test convolution with nonsym symmetry.

void
test_conv_nonsym_d(vsip_support_region support,
                   vsip_length Nr,	// input rows
                   vsip_length Nc,	// input cols
                   vsip_length Mr,	// coeff rows
                   vsip_length Mc,	// coeff cols
                   vsip_index r,
                   vsip_index c,
                   int k1)
{
  vsip_symmetry const symmetry = VSIP_NONSYM;

  vsip_length const D = 1;				// decimation

  vsip_length const Pr = expected_output_size(support, Mr, Nr, D);
  vsip_length const Pc = expected_output_size(support, Mc, Nc, D);

  int shift_r = expected_shift(support, Mr, D);
  int shift_c = expected_shift(support, Mc, D);

  vsip_mview_d *coeff = vsip_mcreate_d(Mr, Mc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mfill_d(0, coeff);
  vsip_mput_d(coeff, r, c, k1);

  vsip_conv2d_d *conv = vsip_conv2d_create_d(coeff, symmetry, Nr, Nc, D,
                                             support, 0, VSIP_ALG_SPACE);
  vsip_conv2d_attr attr;
  vsip_conv2d_getattr_d(conv, &attr);
  test_assert(attr.symm == symmetry);
  test_assert(attr.support == support);

  test_assert(attr.kernel_len.r == Mr);
  test_assert(attr.kernel_len.c == Mc);

  test_assert(attr.data_len.r == Nr);
  test_assert(attr.data_len.c == Nc);

  test_assert(attr.out_len.r == Pr);
  test_assert(attr.out_len.c == Pc);

  
  vsip_mview_d *in = vsip_mcreate_d(Nr, Nc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mview_d *out = vsip_mcreate_d(Pr, Pc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mfill_d(100, out);
  vsip_mview_d *ex = vsip_mcreate_d(Pr, Pc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mfill_d(100, ex);

  init_in_d(in, 0);

  vsip_convolve2d_d(conv, in, out);

  vsip_bool good = 1;
  vsip_index i, j;
  for (i=0; i<Pr; ++i)
    for (j=0; j<Pc; ++j)
    {
      double val;

      if ((int)i + shift_r - (int)r < 0 || i + shift_r - r >= Nr ||
	  (int)j + shift_c - (int)c < 0 || j + shift_c - c >= Nc)
	val = 0;
      else
	val = vsip_mget_d(in, i + shift_r - r, j + shift_c - c);

      vsip_mput_d(ex, i, j, k1 * val);
    }
  double error = merror_db_d(out, ex);
  test_assert(error < ERROR_THRESH);
}

void
test_cconv_nonsym_d(vsip_support_region support,
                    vsip_length Nr,	// input rows
                    vsip_length Nc,	// input cols
                    vsip_length Mr,	// coeff rows
                    vsip_length Mc,	// coeff cols
                    vsip_index r,
                    vsip_index c,
                    int k1)
{
  vsip_symmetry const symmetry = VSIP_NONSYM;

  vsip_length const D = 1;				// decimation

  vsip_length const Pr = expected_output_size(support, Mr, Nr, D);
  vsip_length const Pc = expected_output_size(support, Mc, Nc, D);

  int shift_r = expected_shift(support, Mr, D);
  int shift_c = expected_shift(support, Mc, D);

  vsip_cmview_d *coeff = vsip_cmcreate_d(Mr, Mc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmfill_d(vsip_cmplx_d(0, 0), coeff);
  vsip_cmput_d(coeff, r, c, vsip_cmplx_d(k1, 0));

  vsip_cconv2d_d *conv = vsip_cconv2d_create_d(coeff, symmetry, Nr, Nc, D,
                                               support, 0, VSIP_ALG_SPACE);
  vsip_cconv2d_attr attr;
  vsip_cconv2d_getattr_d(conv, &attr);
  test_assert(attr.symm == symmetry);
  test_assert(attr.support == support);

  test_assert(attr.kernel_len.r == Mr);
  test_assert(attr.kernel_len.c == Mc);

  test_assert(attr.data_len.r == Nr);
  test_assert(attr.data_len.c == Nc);

  test_assert(attr.out_len.r == Pr);
  test_assert(attr.out_len.c == Pc);

  
  vsip_cmview_d *in = vsip_cmcreate_d(Nr, Nc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmview_d *out = vsip_cmcreate_d(Pr, Pc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmfill_d(vsip_cmplx_d(100, 0), out);
  vsip_cmview_d *ex = vsip_cmcreate_d(Pr, Pc, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmfill_d(vsip_cmplx_d(100, 0), ex);

  cinit_in_d(in, 0);

  vsip_cconvolve2d_d(conv, in, out);

  vsip_bool good = 1;
  vsip_index i, j;
  for (i=0; i<Pr; ++i)
    for (j=0; j<Pc; ++j)
    {
      vsip_cscalar_d val;

      if ((int)i + shift_r - (int)r < 0 || i + shift_r - r >= Nr ||
	  (int)j + shift_c - (int)c < 0 || j + shift_c - c >= Nc)
	val.r = val.i = 0;
      else
	val = vsip_cmget_d(in, i + shift_r - r, j + shift_c - c);

      vsip_cmput_d(ex, i, j, vsip_cmplx_d(k1 * val.r, k1 * val.i));
    }
  double error = cmerror_db_d(out, ex);
  test_assert(error < ERROR_THRESH);
}

// Run a set of convolutions for given type and size
//   (with symmetry = nonsym and decimation = 1).

void
cases_nonsym_d(vsip_length i_r,	// input rows
               vsip_length i_c,	// input cols
               vsip_length k_r,	// kernel rows
               vsip_length k_c)	// kernel cols
{
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, i_r, i_c, k_r, k_c,     0,     0, +1);
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, i_r, i_c, k_r, k_c, k_r/2, k_c/2, -2);
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, i_r, i_c, k_r, k_c,     0, k_c-1, +3);
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, i_r, i_c, k_r, k_c, k_r-1,     0, -4);
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, i_r, i_c, k_r, k_c, k_r-1, k_c-1, +5);

  test_conv_nonsym_d(VSIP_SUPPORT_SAME, i_r, i_c, k_r, k_c,     0,     0, +1);
  test_conv_nonsym_d(VSIP_SUPPORT_SAME, i_r, i_c, k_r, k_c, k_r/2, k_c/2, -2);
  test_conv_nonsym_d(VSIP_SUPPORT_SAME, i_r, i_c, k_r, k_c,     0, k_c-1, +3);
  test_conv_nonsym_d(VSIP_SUPPORT_SAME, i_r, i_c, k_r, k_c, k_r-1,     0, -4);
  test_conv_nonsym_d(VSIP_SUPPORT_SAME, i_r, i_c, k_r, k_c, k_r-1, k_c-1, +5);

  test_conv_nonsym_d(VSIP_SUPPORT_FULL, i_r, i_c, k_r, k_c,     0,     0, +1);
  test_conv_nonsym_d(VSIP_SUPPORT_FULL, i_r, i_c, k_r, k_c, k_r/2, k_c/2, -2);
  test_conv_nonsym_d(VSIP_SUPPORT_FULL, i_r, i_c, k_r, k_c,     0, k_c-1, +3);
  test_conv_nonsym_d(VSIP_SUPPORT_FULL, i_r, i_c, k_r, k_c, k_r-1,     0, -4);
  test_conv_nonsym_d(VSIP_SUPPORT_FULL, i_r, i_c, k_r, k_c, k_r-1, k_c-1, +5);
}

void
ccases_nonsym_d(vsip_length i_r,	// input rows
                vsip_length i_c,	// input cols
                vsip_length k_r,	// kernel rows
                vsip_length k_c)	// kernel cols
{
  test_cconv_nonsym_d(VSIP_SUPPORT_MIN, i_r, i_c, k_r, k_c,     0,     0, +1);
  test_cconv_nonsym_d(VSIP_SUPPORT_MIN, i_r, i_c, k_r, k_c, k_r/2, k_c/2, -2);
  test_cconv_nonsym_d(VSIP_SUPPORT_MIN, i_r, i_c, k_r, k_c,     0, k_c-1, +3);
  test_cconv_nonsym_d(VSIP_SUPPORT_MIN, i_r, i_c, k_r, k_c, k_r-1,     0, -4);
  test_cconv_nonsym_d(VSIP_SUPPORT_MIN, i_r, i_c, k_r, k_c, k_r-1, k_c-1, +5);

  test_cconv_nonsym_d(VSIP_SUPPORT_SAME, i_r, i_c, k_r, k_c,     0,     0, +1);
  test_cconv_nonsym_d(VSIP_SUPPORT_SAME, i_r, i_c, k_r, k_c, k_r/2, k_c/2, -2);
  test_cconv_nonsym_d(VSIP_SUPPORT_SAME, i_r, i_c, k_r, k_c,     0, k_c-1, +3);
  test_cconv_nonsym_d(VSIP_SUPPORT_SAME, i_r, i_c, k_r, k_c, k_r-1,     0, -4);
  test_cconv_nonsym_d(VSIP_SUPPORT_SAME, i_r, i_c, k_r, k_c, k_r-1, k_c-1, +5);

  test_cconv_nonsym_d(VSIP_SUPPORT_FULL, i_r, i_c, k_r, k_c,     0,     0, +1);
  test_cconv_nonsym_d(VSIP_SUPPORT_FULL, i_r, i_c, k_r, k_c, k_r/2, k_c/2, -2);
  test_cconv_nonsym_d(VSIP_SUPPORT_FULL, i_r, i_c, k_r, k_c,     0, k_c-1, +3);
  test_cconv_nonsym_d(VSIP_SUPPORT_FULL, i_r, i_c, k_r, k_c, k_r-1,     0, -4);
  test_cconv_nonsym_d(VSIP_SUPPORT_FULL, i_r, i_c, k_r, k_c, k_r-1, k_c-1, +5);
}

int
main(int argc, char** argv)
{
  vsip_init(0);

  // small sets of tests, covered by 'cases()' below
  cases_nonsym_d(8, 8, 3, 3);
/*   ccases_nonsym_d(8, 8, 3, 3); */

  // individual tests, covered by 'cases()' below
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, 8, 8, 3, 3, 1, 1, 1);
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, 8, 8, 3, 3, 0, 2, -1);
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, 8, 8, 3, 3, 2, 0, 2);
  test_conv_nonsym_d(VSIP_SUPPORT_MIN, 8, 8, 3, 3, 2, 2, -2);

  test_conv_nonsym_d(VSIP_SUPPORT_SAME, 8, 8, 3, 3, 1, 1, 1);
  test_conv_nonsym_d(VSIP_SUPPORT_SAME, 8, 8, 3, 3, 0, 2, -1);
  test_conv_nonsym_d(VSIP_SUPPORT_SAME, 8, 8, 3, 3, 2, 0, 2);

  test_conv_nonsym_d(VSIP_SUPPORT_FULL, 8, 8, 3, 3, 1, 1, 1);
  test_conv_nonsym_d(VSIP_SUPPORT_FULL, 8, 8, 3, 3, 0, 0, 2);
  test_conv_nonsym_d(VSIP_SUPPORT_FULL, 8, 8, 3, 3, 2, 2, -1);
  return 0;
}

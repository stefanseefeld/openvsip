//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is made available under the GPL.
// See the accompanying file LICENSE.GPL for details.

#include <vsip.h>
#include "output.h"
#include "test.h"

#define DEBUG 1

void
test_complex_d(vsip_length N, vsip_major axis)
{
  vsip_length rows = 5;
  vsip_length cols = N;
  if (axis == VSIP_COL)
  {
    rows = N;
    cols = 5;
  }
  vsip_fftm_d *f_fftm = vsip_ccfftmop_create_d(rows, cols, 1.0,
                                               VSIP_FFT_FWD, axis, 0, VSIP_ALG_SPACE);
  vsip_fftm_d *i_fftm = vsip_ccfftmop_create_d(rows, cols, 1.0/N,
                                               VSIP_FFT_INV, axis, 0, VSIP_ALG_SPACE);

  vsip_fftm_attr_d attr;
  vsip_fftm_getattr_d(f_fftm, &attr);
  test_assert(attr.input.r == rows && attr.input.c == cols);
  test_assert(attr.output.r == rows && attr.output.c == cols);
  vsip_fftm_getattr_d(i_fftm, &attr);
  test_assert(attr.input.r == rows && attr.input.c == cols);
  test_assert(attr.output.r == rows && attr.output.c == cols);

  vsip_cmview_d *in = vsip_cmcreate_d(rows, cols, VSIP_ROW, VSIP_MEM_NONE);
  vsip_randstate *rng = vsip_randcreate(0, 1, 1, VSIP_PRNG);
  vsip_cmrandu_d(rng, in);
  vsip_randdestroy(rng);
  vsip_cmview_d *out = vsip_cmcreate_d(rows, cols, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmview_d *inv = vsip_cmcreate_d(rows, cols, VSIP_ROW, VSIP_MEM_NONE);
  vsip_ccfftmop_d(f_fftm, in, out);
  vsip_ccfftmop_d(i_fftm, out, inv);
  test_assert(cmequal_d(inv, in));
  vsip_cmalldestroy_d(inv);
  vsip_cmalldestroy_d(out);
  vsip_cmalldestroy_d(in);
  vsip_fftm_destroy_d(i_fftm);
  vsip_fftm_destroy_d(f_fftm);
}

void
test_real_d(vsip_length N, vsip_major axis)
{
  vsip_length rows = 5;
  vsip_length cols = N;
  vsip_length rows2 = 5;
  vsip_length cols2 = N/2 + 1;
  if (axis == VSIP_COL)
  {
    rows = N;
    cols = 5;
    rows2 = N/2 + 1;
    cols2 = 5;
  }
  vsip_fftm_d *f_fftm = vsip_rcfftmop_create_d(rows, cols, 1.0,
                                               axis, 0, VSIP_ALG_SPACE);
  vsip_fftm_d *i_fftm = vsip_crfftmop_create_d(rows, cols, 1.0/N,
                                               axis, 0, VSIP_ALG_SPACE);

  vsip_fftm_attr_d attr;
  vsip_fftm_getattr_d(f_fftm, &attr);
  test_assert(attr.input.r == rows && attr.input.c == cols);
  test_assert(attr.output.r == rows2 && attr.output.c == cols2);

  vsip_fftm_getattr_d(i_fftm, &attr);
  test_assert(attr.input.r == rows2 && attr.input.c == cols2);
  test_assert(attr.output.r == rows && attr.output.c == cols);

  vsip_mview_d *in = vsip_mcreate_d(rows, cols, VSIP_ROW, VSIP_MEM_NONE);
  vsip_randstate *rng = vsip_randcreate(0, 1, 1, VSIP_PRNG);
  vsip_mrandu_d(rng, in);
  vsip_randdestroy(rng);
  vsip_mview_d *inv = vsip_mcreate_d(rows, cols, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmview_d *out = vsip_cmcreate_d(rows2, cols2, VSIP_ROW, VSIP_MEM_NONE);

  vsip_rcfftmop_d(f_fftm, in, out);
  vsip_crfftmop_d(i_fftm, out, inv);
  test_assert(merror_db_d(inv, in) < -100.);
  vsip_malldestroy_d(inv);
  vsip_cmalldestroy_d(out);
  vsip_malldestroy_d(in);
  vsip_fftm_destroy_d(i_fftm);
  vsip_fftm_destroy_d(f_fftm);
}

int
main(int argc, char** argv)
{
  vsip_init(0);

  test_complex_d(128, VSIP_ROW);
  test_complex_d(256, VSIP_ROW);
  test_complex_d(512, VSIP_ROW);

  test_complex_d(18, VSIP_COL);
  test_complex_d(256, VSIP_COL);

  test_real_d(242, VSIP_ROW);
  test_real_d(128, VSIP_ROW);
  test_real_d(16, VSIP_ROW);

  test_real_d(242, VSIP_COL);
  test_real_d(128, VSIP_COL);
  test_real_d(16, VSIP_COL);

  return 0;
}

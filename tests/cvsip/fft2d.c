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
test_complex_d(vsip_length rows, vsip_length cols, vsip_major axis, vsip_fft_dir dir)
{
  // Use this identity for validation: FFT2D == FFTM(rows) * FFTM(cols)

  vsip_fftm_d *r_fftm = vsip_ccfftmop_create_d(rows, cols, 1.0,
                                               dir, VSIP_ROW, 0, VSIP_ALG_SPACE);
  vsip_fftm_d *c_fftm = vsip_ccfftmop_create_d(rows, cols, 1.0,
                                               dir, VSIP_COL, 0, VSIP_ALG_SPACE);

  vsip_fft2d_d *fft = vsip_ccfft2dop_create_d(rows, cols, 1.0, dir, 0, VSIP_ALG_SPACE);
  vsip_fft2d_attr_d attr;
  vsip_fft2d_getattr_d(fft, &attr);
  test_assert(attr.input.r == rows);
  test_assert(attr.input.c == cols);
  test_assert(attr.output.r == rows);
  test_assert(attr.output.c == cols);
  test_assert(attr.place == VSIP_FFT_OP);
  test_assert(attr.scale == 1.0);
  test_assert(attr.dir == dir);

  vsip_cmview_d *in = vsip_cmcreate_d(rows, cols, axis, VSIP_MEM_NONE);
  vsip_randstate *rng = vsip_randcreate(0, 1, 1, VSIP_PRNG);
  vsip_cmrandu_d(rng, in);
  vsip_randdestroy(rng);
  vsip_cmview_d *out = vsip_cmcreate_d(rows, cols, axis, VSIP_MEM_NONE);
  vsip_cmview_d *ref = vsip_cmcreate_d(rows, cols, axis, VSIP_MEM_NONE);

  vsip_ccfft2dop_d(fft, in, out);
  vsip_ccfftmop_d(r_fftm, in, ref);
  vsip_ccfftmip_d(c_fftm, ref);
  test_assert(cmerror_db_d(out, ref) < -100.);

  vsip_cmalldestroy_d(ref);
  vsip_cmalldestroy_d(out);
  vsip_cmalldestroy_d(in);
}

void
test_real_d(vsip_length rows, vsip_length cols, vsip_major axis, vsip_fft_dir dir)
{
  // Use this identity for validation: FFT2D == FFTM(rows) * FFTM(cols)

  vsip_fftm_d *r_fftm;
  vsip_fftm_d *c_fftm;
  vsip_fft2d_d *fft;
  vsip_major minor = axis == VSIP_COL ? VSIP_ROW : VSIP_COL;
  vsip_randstate *rng = vsip_randcreate(0, 1, 1, VSIP_PRNG);
  if (dir == VSIP_FFT_FWD)
  {
    fft = vsip_rcfft2dop_create_d(rows, cols, 1.0, 0, VSIP_ALG_SPACE);
    r_fftm = vsip_rcfftmop_create_d(rows, cols, 1.0,
                                    axis, 0, VSIP_ALG_SPACE);
  }
  else
  {
    fft = vsip_crfft2dop_create_d(rows, cols, 1.0, 0, VSIP_ALG_SPACE);
    r_fftm = vsip_crfftmop_create_d(rows, cols, 1.0,
                                    axis, 0, VSIP_ALG_SPACE);
  }
  if (axis == VSIP_ROW)
    c_fftm = vsip_ccfftmip_create_d(rows, cols/2+1, 1.0,
                                    dir, minor, 0, VSIP_ALG_SPACE);
  else // axis == VSIP_COL
    c_fftm = vsip_ccfftmip_create_d(rows/2+1, cols, 1.0,
                                    dir, minor, 0, VSIP_ALG_SPACE);

#if 0
  vsip_fft2d_attr_d attr;
  vsip_fft2d_getattr_d(fft, &attr);
  test_assert(attr.input.r == rows);
  test_assert(attr.input.c == cols);
  test_assert(attr.output.r == rows);
  test_assert(attr.output.c == cols);
  test_assert(attr.place == VSIP_FFT_OP);
  test_assert(attr.scale == 1.0);
  test_assert(attr.dir == dir);
#endif

  if (dir == VSIP_FFT_FWD)
  {
    vsip_mview_d *in = vsip_mcreate_d(rows, cols, axis, VSIP_MEM_NONE);
    vsip_mrandu_d(rng, in);
    if (axis == VSIP_ROW) cols = cols/2 + 1;
    else rows = rows/2 + 1;
    vsip_cmview_d *out = vsip_cmcreate_d(rows, cols, axis, VSIP_MEM_NONE);
    vsip_cmview_d *ref = vsip_cmcreate_d(rows, cols, axis, VSIP_MEM_NONE);

    vsip_rcfft2dop_d(fft, in, out);
    vsip_rcfftmop_d(r_fftm, in, ref);
    vsip_ccfftmip_d(c_fftm, ref);
    test_assert(cmerror_db_d(out, ref) < -100.);

    vsip_cmalldestroy_d(ref);
    vsip_cmalldestroy_d(out);
    vsip_malldestroy_d(in);
  }

  else // dir == VISP_FFT_INV
  {
    vsip_mview_d *out = vsip_mcreate_d(rows, cols, axis, VSIP_MEM_NONE);
    vsip_mview_d *ref = vsip_mcreate_d(rows, cols, axis, VSIP_MEM_NONE);
    if (axis == VSIP_ROW) cols = cols/2 + 1;
    else rows = rows/2 + 1;
    vsip_cmview_d *in = vsip_cmcreate_d(rows, cols, axis, VSIP_MEM_NONE);
    vsip_cmrandu_d(rng, in);

    vsip_crfft2dop_d(fft, in, out);
    vsip_ccfftmip_d(c_fftm, in);
    vsip_crfftmop_d(r_fftm, in, ref);
    test_assert(merror_db_d(out, ref) < -100.);

    vsip_cmalldestroy_d(in);
    vsip_malldestroy_d(ref);
    vsip_malldestroy_d(out);
  }

  vsip_fft2d_destroy_d(fft);
  vsip_fftm_destroy_d(c_fftm);
  vsip_fftm_destroy_d(r_fftm);
  vsip_randdestroy(rng);
}

int
main(int argc, char **argv)
{
  vsip_init(0);

  test_complex_d(128, 128, VSIP_ROW, VSIP_FFT_FWD);
  test_complex_d(512, 128, VSIP_ROW, VSIP_FFT_FWD);
  test_complex_d(128, 512, VSIP_ROW, VSIP_FFT_FWD);

  test_complex_d(128, 128, VSIP_COL, VSIP_FFT_FWD);
  test_complex_d(512, 128, VSIP_COL, VSIP_FFT_FWD);
  test_complex_d(128, 512, VSIP_COL, VSIP_FFT_FWD);

  test_complex_d(128, 128, VSIP_ROW, VSIP_FFT_INV);
  test_complex_d(512, 128, VSIP_ROW, VSIP_FFT_INV);
  test_complex_d(128, 512, VSIP_ROW, VSIP_FFT_INV);

  test_complex_d(128, 128, VSIP_COL, VSIP_FFT_INV);
  test_complex_d(512, 128, VSIP_COL, VSIP_FFT_INV);
  test_complex_d(128, 512, VSIP_COL, VSIP_FFT_INV);

  test_real_d(128, 128, VSIP_ROW, VSIP_FFT_FWD);
  test_real_d(512, 128, VSIP_ROW, VSIP_FFT_FWD);
  test_real_d(128, 512, VSIP_ROW, VSIP_FFT_FWD);

  test_real_d(128, 128, VSIP_COL, VSIP_FFT_FWD);
  test_real_d(512, 128, VSIP_COL, VSIP_FFT_FWD);
  test_real_d(128, 512, VSIP_COL, VSIP_FFT_FWD);

  test_real_d(128, 128, VSIP_ROW, VSIP_FFT_INV);
  test_real_d(512, 128, VSIP_ROW, VSIP_FFT_INV);
  test_real_d(128, 512, VSIP_ROW, VSIP_FFT_INV);

  test_real_d(128, 128, VSIP_COL, VSIP_FFT_INV);
  test_real_d(512, 128, VSIP_COL, VSIP_FFT_INV);
  test_real_d(128, 512, VSIP_COL, VSIP_FFT_INV);

  return 0;
}

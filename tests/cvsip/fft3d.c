/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    tests/cvsip/fft3d.c
    @author  Stefan Seefeld
    @date    2008-07-07
    @brief   test 3d fft operations.
*/
#include <vsip.h>
#include "output.h"
#include "test.h"

#define DEBUG 1

void
test_complex_d(vsip_length z, vsip_length y, vsip_length x,
               vsip_fft_dir dir)
{
  vsip_fft_d *z_fft = vsip_ccfftop_create_d(z, 1.0, dir, 0, VSIP_ALG_SPACE);
  vsip_fft_d *y_fft = vsip_ccfftip_create_d(y, 1.0, dir, 0, VSIP_ALG_SPACE);
  vsip_fft_d *x_fft = vsip_ccfftip_create_d(x, 1.0, dir, 0, VSIP_ALG_SPACE);
  vsip_fft3d_d *fft = vsip_ccfft3dop_create_d(z, y, x, 1.0, dir, 0, VSIP_ALG_SPACE);

  vsip_fft3d_attr_d attr;
  vsip_fft3d_getattr_d(fft, &attr);
  test_assert(attr.input.z == z);
  test_assert(attr.input.y == y);
  test_assert(attr.input.x == x);
  test_assert(attr.output.z == z);
  test_assert(attr.output.y == y);
  test_assert(attr.output.x == x);
  test_assert(attr.place == VSIP_FFT_OP);
  test_assert(attr.scale == 1.0);
  test_assert(attr.dir == dir);

  vsip_ctview_d *in = vsip_ctcreate_d(z, y, x, VSIP_TRAILING, VSIP_MEM_NONE);
  vsip_randstate *rng = vsip_randcreate(0, 1, 1, VSIP_PRNG);
  vsip_ctrandu_d(rng, in);
  vsip_randdestroy(rng);
  vsip_ctview_d *out = vsip_ctcreate_d(z, y, x, VSIP_TRAILING, VSIP_MEM_NONE);
  vsip_ctview_d *ref = vsip_ctcreate_d(z, y, x, VSIP_TRAILING, VSIP_MEM_NONE);

  vsip_ccfft3dop_d(fft, in, out);

  size_t i, j;

  for (i = 0; i != z; ++i)
    for (j = 0; j != y; ++j)
    {
      vsip_cvview_d *in_sub = vsip_ctvectview_d(in, VSIP_TVX, i, j);
      vsip_cvview_d *ref_sub = vsip_ctvectview_d(ref, VSIP_TVX, i, j);
      vsip_ccfftop_d(x_fft, in_sub, ref_sub);
      vsip_cvdestroy_d(ref_sub);
      vsip_cvdestroy_d(in_sub);
    }
  for (i = 0; i != z; ++i)
    for (j = 0; j != x; ++j)
    {
      vsip_cvview_d *sub = vsip_ctvectview_d(ref, VSIP_TVY, i, j);
      vsip_ccfftip_d(y_fft, sub);
      vsip_cvdestroy_d(sub);
    }
  for (i = 0; i != y; ++i)
    for (j = 0; j != x; ++j)
    {
      vsip_cvview_d *sub = vsip_ctvectview_d(ref, VSIP_TVZ, i, j);
      vsip_ccfftip_d(z_fft, sub);
      vsip_cvdestroy_d(sub);
    }

  test_assert(cterror_db_d(out, ref)  < -100.);

  vsip_ctalldestroy_d(ref);
  vsip_ctalldestroy_d(out);
  vsip_ctalldestroy_d(in);
}

void
test_real_d(vsip_length z, vsip_length y, vsip_length x,
            int axis, vsip_fft_dir dir)
{
  vsip_fft_d *z_fft;
  vsip_fft_d *y_fft;
  vsip_fft_d *x_fft;
  if (axis == 0)
  {
    if (dir == VSIP_FFT_FWD)
      z_fft = vsip_rcfftop_create_d(z, 1.0, 0, VSIP_ALG_SPACE);
    else
      z_fft = vsip_crfftop_create_d(z, 1.0, 0, VSIP_ALG_SPACE);
    y_fft = vsip_ccfftip_create_d(y, 1.0, dir, 0, VSIP_ALG_SPACE);
    x_fft = vsip_ccfftip_create_d(x, 1.0, dir, 0, VSIP_ALG_SPACE);
  }
  else if (axis == 1)
  {
    z_fft = vsip_ccfftip_create_d(z, 1.0, dir, 0, VSIP_ALG_SPACE);
    if (dir == VSIP_FFT_FWD)
      y_fft = vsip_rcfftop_create_d(y, 1.0, 0, VSIP_ALG_SPACE);
    else
      y_fft = vsip_crfftop_create_d(y, 1.0, 0, VSIP_ALG_SPACE);
    x_fft = vsip_ccfftip_create_d(x, 1.0, dir, 0, VSIP_ALG_SPACE);
  }
  else
  {
    z_fft = vsip_ccfftip_create_d(z, 1.0, dir, 0, VSIP_ALG_SPACE);
    y_fft = vsip_ccfftip_create_d(y, 1.0, dir, 0, VSIP_ALG_SPACE);
    if (dir == VSIP_FFT_FWD)
      x_fft = vsip_rcfftop_create_d(x, 1.0, 0, VSIP_ALG_SPACE);
    else
      x_fft = vsip_crfftop_create_d(x, 1.0, 0, VSIP_ALG_SPACE);
  }
  vsip_fft3d_d *fft;
  if (dir == VSIP_FFT_FWD)
    fft = vsip_rcfft3dop_create_d(z, y, x, 1.0, 0, VSIP_ALG_SPACE);
  else
    fft = vsip_crfft3dop_create_d(z, y, x, 1.0, 0, VSIP_ALG_SPACE);

#if 0
  vsip_fft3d_attr_d attr;
  vsip_fft3d_getattr_d(fft, &attr);
  test_assert(attr.input.z == z);
  test_assert(attr.input.y == y);
  test_assert(attr.input.x == x);
  test_assert(attr.output.z == z);
  test_assert(attr.output.y == y);
  test_assert(attr.output.x == x);
  test_assert(attr.place == VSIP_FFT_OP);
  test_assert(attr.scale == 1.0);
  test_assert(attr.dir == dir);
#endif

  if (dir == VSIP_FFT_FWD)
  {
    size_t z_out = axis == 0 ? z/2+1 : z;
    size_t y_out = axis == 1 ? y/2+1 : y;
    size_t x_out = axis == 2 ? x/2+1 : x;
    vsip_tview_d *in_raw = vsip_tcreate_d(z, y, x, VSIP_TRAILING, VSIP_MEM_NONE);
    vsip_tview_d *in;
    if (axis == 0) in = vsip_ttransview_d(in_raw, VSIP_TTRANS_ZX);
    else if (axis == 1) in = vsip_ttransview_d(in_raw, VSIP_TTRANS_YX);
    else in = vsip_ttransview_d(in_raw, VSIP_TTRANS_NOP);

    vsip_randstate *rng = vsip_randcreate(0, 1, 1, VSIP_PRNG);
    vsip_trandu_d(rng, in);
    vsip_randdestroy(rng);
    vsip_ctview_d *out = vsip_ctcreate_d(z_out, y_out, x_out, VSIP_TRAILING, VSIP_MEM_NONE);
    vsip_ctview_d *ref = vsip_ctcreate_d(z_out, y_out, x_out, VSIP_TRAILING, VSIP_MEM_NONE);

    vsip_rcfft3dop_d(fft, in, out);

    size_t i, j;

    if (axis == 2)
    {
      for (i = 0; i != z_out; ++i)
        for (j = 0; j != y_out; ++j)
        {
          vsip_vview_d *in_sub = vsip_tvectview_d(in, VSIP_TVX, i, j);
          vsip_cvview_d *ref_sub = vsip_ctvectview_d(ref, VSIP_TVX, i, j);
          vsip_rcfftop_d(x_fft, in_sub, ref_sub);
          vsip_cvdestroy_d(ref_sub);
          vsip_vdestroy_d(in_sub);
        }
      for (i = 0; i != z_out; ++i)
        for (j = 0; j != x_out; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(ref, VSIP_TVY, i, j);
          vsip_ccfftip_d(y_fft, sub);
          vsip_cvdestroy_d(sub);
        }
      for (i = 0; i != y_out; ++i)
        for (j = 0; j != x_out; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(ref, VSIP_TVZ, i, j);
          vsip_ccfftip_d(z_fft, sub);
          vsip_cvdestroy_d(sub);
        }
    }
    else if (axis == 1)
    {
      for (i = 0; i != z_out; ++i)
        for (j = 0; j != x_out; ++j)
        {
          vsip_vview_d *in_sub = vsip_tvectview_d(in, VSIP_TVY, i, j);
          vsip_cvview_d *ref_sub = vsip_ctvectview_d(ref, VSIP_TVY, i, j);
          vsip_rcfftop_d(y_fft, in_sub, ref_sub);
          vsip_cvdestroy_d(ref_sub);
          vsip_vdestroy_d(in_sub);
        }
      for (i = 0; i != z_out; ++i)
        for (j = 0; j != y_out; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(ref, VSIP_TVX, i, j);
          vsip_ccfftip_d(x_fft, sub);
          vsip_cvdestroy_d(sub);
        }
      for (i = 0; i != y_out; ++i)
        for (j = 0; j != x_out; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(ref, VSIP_TVZ, i, j);
          vsip_ccfftip_d(z_fft, sub);
          vsip_cvdestroy_d(sub);
        }
    }
    else // axis == 0
    {
      for (i = 0; i != y_out; ++i)
        for (j = 0; j != x_out; ++j)
        {
          vsip_vview_d *in_sub = vsip_tvectview_d(in, VSIP_TVZ, i, j);
          vsip_cvview_d *ref_sub = vsip_ctvectview_d(ref, VSIP_TVZ, i, j);
          vsip_rcfftop_d(z_fft, in_sub, ref_sub);
          vsip_cvdestroy_d(ref_sub);
          vsip_vdestroy_d(in_sub);
        }
      for (i = 0; i != z_out; ++i)
        for (j = 0; j != x_out; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(ref, VSIP_TVY, i, j);
          vsip_ccfftip_d(y_fft, sub);
          vsip_cvdestroy_d(sub);
        }
      for (i = 0; i != z_out; ++i)
        for (j = 0; j != y_out; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(ref, VSIP_TVX, i, j);
          vsip_ccfftip_d(x_fft, sub);
          vsip_cvdestroy_d(sub);
        }
    }

    test_assert(cterror_db_d(out, ref)  < -100.);

    vsip_ctalldestroy_d(ref);
    vsip_ctalldestroy_d(out);
    vsip_tdestroy_d(in);
    vsip_talldestroy_d(in_raw);
  }
  else // dir == VSIP_FFT_INV
  {
    // To obtain the unit-stride along the desired axis, we transpose the tensor.
    vsip_ctview_d *in_raw = vsip_ctcreate_d(z, y, x/2 + 1, VSIP_TRAILING, VSIP_MEM_NONE);
    vsip_ctview_d *in;
    if (axis == 0) in = vsip_cttransview_d(in_raw, VSIP_TTRANS_ZX);
    else if (axis == 1) in = vsip_cttransview_d(in_raw, VSIP_TTRANS_YX);
    else in = vsip_cttransview_d(in_raw, VSIP_TTRANS_NOP);
    vsip_randstate *rng = vsip_randcreate(0, 1, 1, VSIP_PRNG);
    vsip_ctrandu_d(rng, in);
    vsip_randdestroy(rng);
    vsip_tview_d *out = vsip_tcreate_d(z, y, x, VSIP_TRAILING, VSIP_MEM_NONE);
    vsip_tview_d *ref = vsip_tcreate_d(z, y, x, VSIP_TRAILING, VSIP_MEM_NONE);

    vsip_crfft3dop_d(fft, in, out);

    size_t i, j;

    if (axis == 2)
    {
      for (i = 0; i != z; ++i)
        for (j = 0; j != x/2+1; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(in, VSIP_TVY, i, j);
          vsip_ccfftip_d(y_fft, sub);
          vsip_cvdestroy_d(sub);
        }
      for (i = 0; i != y; ++i)
        for (j = 0; j != x/2+1; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(in, VSIP_TVZ, i, j);
          vsip_ccfftip_d(z_fft, sub);
          vsip_cvdestroy_d(sub);
        }
      for (i = 0; i != z; ++i)
        for (j = 0; j != y; ++j)
        {
          vsip_cvview_d *in_sub = vsip_ctvectview_d(in, VSIP_TVX, i, j);
          vsip_vview_d *ref_sub = vsip_tvectview_d(ref, VSIP_TVX, i, j);
          vsip_crfftop_d(x_fft, in_sub, ref_sub);
          vsip_vdestroy_d(ref_sub);
          vsip_cvdestroy_d(in_sub);
        }
    }
    else if (axis == 1)
    {
      for (i = 0; i != z; ++i)
        for (j = 0; j != y/2+1; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(in, VSIP_TVX, i, j);
          vsip_ccfftip_d(x_fft, sub);
          vsip_cvdestroy_d(sub);
        }
      for (i = 0; i != y/2+1; ++i)
        for (j = 0; j != x; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(in, VSIP_TVZ, i, j);
          vsip_ccfftip_d(z_fft, sub);
          vsip_cvdestroy_d(sub);
        }
      for (i = 0; i != z; ++i)
        for (j = 0; j != x; ++j)
        {
          vsip_cvview_d *in_sub = vsip_ctvectview_d(in, VSIP_TVY, i, j);
          vsip_vview_d *ref_sub = vsip_tvectview_d(ref, VSIP_TVY, i, j);
          vsip_crfftop_d(y_fft, in_sub, ref_sub);
          vsip_vdestroy_d(ref_sub);
          vsip_cvdestroy_d(in_sub);
        }
    }
    else // axis == 0
    {
      for (i = 0; i != z/2+1; ++i)
        for (j = 0; j != x; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(in, VSIP_TVY, i, j);
          vsip_ccfftip_d(y_fft, sub);
          vsip_cvdestroy_d(sub);
        }
      for (i = 0; i != z/2+1; ++i)
        for (j = 0; j != y; ++j)
        {
          vsip_cvview_d *sub = vsip_ctvectview_d(in, VSIP_TVX, i, j);
          vsip_ccfftip_d(x_fft, sub);
          vsip_cvdestroy_d(sub);
        }
      for (i = 0; i != y; ++i)
        for (j = 0; j != x; ++j)
        {
          vsip_cvview_d *in_sub = vsip_ctvectview_d(in, VSIP_TVZ, i, j);
          vsip_vview_d *ref_sub = vsip_tvectview_d(ref, VSIP_TVZ, i, j);
          vsip_crfftop_d(z_fft, in_sub, ref_sub);
          vsip_vdestroy_d(ref_sub);
          vsip_cvdestroy_d(in_sub);
        }
    }

    test_assert(terror_db_d(out, ref)  < -100.);

    vsip_talldestroy_d(ref);
    vsip_talldestroy_d(out);
    vsip_ctdestroy_d(in);
    vsip_ctalldestroy_d(in_raw);
  }
}


int
main(int argc, char **argv)
{
  vsip_init(0);

  test_complex_d(32, 32, 32, VSIP_FFT_FWD);
  test_complex_d(64, 32, 32, VSIP_FFT_FWD);
  test_complex_d(32, 64, 32, VSIP_FFT_FWD);

  test_complex_d(32, 32, 32, VSIP_FFT_INV);
  test_complex_d(64, 32, 32, VSIP_FFT_INV);
  test_complex_d(32, 64, 32, VSIP_FFT_INV);

  test_real_d(32, 32, 32, 0, VSIP_FFT_FWD);
  test_real_d(32, 32, 32, 1, VSIP_FFT_FWD);
  test_real_d(32, 32, 32, 2, VSIP_FFT_FWD);

  test_real_d(32, 32, 32, 0, VSIP_FFT_INV);
  test_real_d(32, 32, 32, 1, VSIP_FFT_INV);
  test_real_d(32, 32, 32, 2, VSIP_FFT_INV);

  return 0;
}

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

#define VERBOSE 1

/***********************************************************************
  Test IIR as single FIR -- no recursion
***********************************************************************/

void
iir_as_fir_case_d(vsip_length size,
                  vsip_length chunk,
                  vsip_mview_d *b,
                  vsip_obj_state state)
{
  vsip_length order = vsip_mgetcollength_d(b);

  vsip_mview_d *a = vsip_mcreate_d(order, 2, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mfill_d(0, a);

  test_assert(vsip_mgetrowlength_d(b) == 3);	// IIR is only 2nd order

  test_assert(chunk <= size);
  test_assert(size % chunk == 0);

  vsip_iir_d *iir = vsip_iir_create_d(b, a, chunk, state, 0, VSIP_ALG_SPACE);
  vsip_iir_attr_d attr;
  vsip_iir_getattr_d(iir, &attr);
  test_assert(attr.n2nd  == 2*order);
  test_assert(attr.seg_len   == chunk);
  test_assert(attr.state == state);

  vsip_fir_d **fir = (vsip_fir_d**)malloc(order * sizeof(vsip_fir_d*));
  vsip_vview_d **rows = (vsip_vview_d**)malloc(order * sizeof(vsip_vview_d*));
  vsip_length m;
  for (m=0; m<order; ++m)
  {
    rows[m] = vsip_mrowview_d(b, m);
    fir[m] = vsip_fir_create_d(rows[m], VSIP_NONSYM, chunk, 1, state, 0, VSIP_ALG_SPACE);
  }
  vsip_vview_d *data = vsip_vcreate_d(size, VSIP_MEM_NONE);
  vsip_vview_d *out_iir = vsip_vcreate_d(size, VSIP_MEM_NONE);
  vsip_vview_d *out_fir = vsip_vcreate_d(size, VSIP_MEM_NONE);
  vsip_vview_d *tmp = vsip_vcreate_d(chunk, VSIP_MEM_NONE);

  vsip_vramp_d(1., 1., data);

  vsip_index pos = 0;
  while (pos < size)
  {
    vsip_vview_d *in = vsip_vsubview_d(data, pos, chunk);
    vsip_vview_d *out = vsip_vsubview_d(out_iir, pos, chunk);
    printf("in :\n");
    voutput_d(in);
    vsip_iirflt_d(iir, in, out);

    vsip_vcopy_d_d(in, tmp);

    vsip_index m;
    for (m=0; m<order; ++m)
    {
      vsip_vview_d *out2 = vsip_vsubview_d(out_fir, pos, chunk);
      vsip_firflt_d(fir[m], tmp, out2);
      vsip_vcopy_d_d(out2, tmp);
      vsip_vdestroy_d(out2);
    }
    pos += chunk;
    vsip_vdestroy_d(out);
    vsip_vdestroy_d(in);
  }

  double error = verror_db_d(out_iir, out_fir);

#if VERBOSE
  if (error >= -150)
  {
    printf("iir =\n");
    voutput_d(out_iir);
    printf("fir =\n");
    voutput_d(out_fir);
  }
#endif

  test_assert(error < -150);

  vsip_iir_destroy_d(iir);
  for (m=0; m<order; ++m)
  {
    vsip_vdestroy_d(rows[m]);
    vsip_fir_destroy_d(fir[m]);
  }
  free(fir);
  free(rows);
  vsip_valldestroy_d(tmp);
  vsip_valldestroy_d(out_fir);
  vsip_valldestroy_d(out_iir);
  vsip_valldestroy_d(data);
}

void
ciir_as_fir_case_d(vsip_length size,
                   vsip_length chunk,
                   vsip_cmview_d *b,
                   vsip_obj_state state)
{
  vsip_length order = vsip_cmgetcollength_d(b);

  vsip_cmview_d *a = vsip_cmcreate_d(order, 2, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmfill_d(vsip_cmplx_d(0, 0), a);

  test_assert(vsip_cmgetrowlength_d(b) == 3);	// IIR is only 2nd order

  test_assert(chunk <= size);
  test_assert(size % chunk == 0);

  vsip_ciir_d *iir = vsip_ciir_create_d(b, a, chunk, state, 0, VSIP_ALG_SPACE);
  vsip_ciir_attr_d attr;
  vsip_ciir_getattr_d(iir, &attr);
  test_assert(attr.n2nd  == 2*order);
  test_assert(attr.seg_len   == chunk);
  test_assert(attr.state == state);

  vsip_cfir_d **fir = (vsip_cfir_d**)malloc(order * sizeof(vsip_cfir_d*));
  vsip_cvview_d **rows = (vsip_cvview_d**)malloc(order * sizeof(vsip_cvview_d*));
  vsip_length m;
  for (m=0; m<order; ++m)
  {
    rows[m] = vsip_cmrowview_d(b, m);
    fir[m] = vsip_cfir_create_d(rows[m], VSIP_NONSYM, chunk, 1, state, 0, VSIP_ALG_SPACE);
  }
  vsip_cvview_d *data = vsip_cvcreate_d(size, VSIP_MEM_NONE);
  vsip_cvview_d *out_iir = vsip_cvcreate_d(size, VSIP_MEM_NONE);
  vsip_cvview_d *out_fir = vsip_cvcreate_d(size, VSIP_MEM_NONE);
  vsip_cvview_d *tmp = vsip_cvcreate_d(chunk, VSIP_MEM_NONE);

  vsip_cvramp_d(vsip_cmplx_d(1., 0.), vsip_cmplx_d(1., 0.), data);

  vsip_index pos = 0;
  while (pos < size)
  {
    vsip_cvview_d *in = vsip_cvsubview_d(data, pos, chunk);
    vsip_cvview_d *out = vsip_cvsubview_d(out_iir, pos, chunk);
    printf("in :\n");
    cvoutput_d(in);
    vsip_ciirflt_d(iir, in, out);

    vsip_cvcopy_d_d(in, tmp);

    vsip_index m;
    for (m=0; m<order; ++m)
    {
      vsip_cvview_d *out2 = vsip_cvsubview_d(out_fir, pos, chunk);
      vsip_cfirflt_d(fir[m], tmp, out2);
      vsip_cvcopy_d_d(out2, tmp);
      vsip_cvdestroy_d(out2);
    }
    pos += chunk;
    vsip_cvdestroy_d(out);
    vsip_cvdestroy_d(in);
  }

  double error = cverror_db_d(out_iir, out_fir);

#if VERBOSE
  if (error >= -150)
  {
    printf("iir =\n");
    cvoutput_d(out_iir);
    printf("fir =\n");
    cvoutput_d(out_fir);
  }
#endif

  test_assert(error < -150);

  vsip_ciir_destroy_d(iir);
  for (m=0; m<order; ++m)
  {
    vsip_cvdestroy_d(rows[m]);
    vsip_cfir_destroy_d(fir[m]);
  }
  free(fir);
  free(rows);
  vsip_cvalldestroy_d(tmp);
  vsip_cvalldestroy_d(out_fir);
  vsip_cvalldestroy_d(out_iir);
  vsip_cvalldestroy_d(data);
}


void
test_iir_as_fir_d()
{
  double d[] = {1, -2, 3,
                3, -1, 1,
                1, 0, -1,
                -1, 2, -2};
  vsip_block_d *b = vsip_blockbind_d(d, 12, VSIP_MEM_NONE);
  vsip_blockadmit_d(b, 1);
  vsip_mview_d *w = vsip_mbind_d(b, 0, 3, 4, 1, 3);

  vsip_length size = 128;

  iir_as_fir_case_d(size, size, w, VSIP_STATE_SAVE);
  iir_as_fir_case_d(size, size/2, w, VSIP_STATE_SAVE);
  iir_as_fir_case_d(size, size/4, w, VSIP_STATE_SAVE);

  iir_as_fir_case_d(size, size, w, VSIP_STATE_NO_SAVE);
  iir_as_fir_case_d(size, size/2, w, VSIP_STATE_NO_SAVE);
  iir_as_fir_case_d(size, size/4, w, VSIP_STATE_NO_SAVE);
}

void
test_ciir_as_fir_d()
{
  double d_re[] = {1, -2, 3,
                   3, -1, 1,
                   1, 0, -1,
                   -1, 2, -2};
  double d_im[12] = {0};
  vsip_cblock_d *b = vsip_cblockbind_d(d_re, d_im, 12, VSIP_MEM_NONE);
  vsip_cblockadmit_d(b, 1);
  vsip_cmview_d *w = vsip_cmbind_d(b, 0, 3, 4, 1, 3);

  vsip_length size = 128;

  ciir_as_fir_case_d(size, size, w, VSIP_STATE_SAVE);
  ciir_as_fir_case_d(size, size/2, w, VSIP_STATE_SAVE);
  ciir_as_fir_case_d(size, size/4, w, VSIP_STATE_SAVE);

  ciir_as_fir_case_d(size, size, w, VSIP_STATE_NO_SAVE);
  ciir_as_fir_case_d(size, size/2, w, VSIP_STATE_NO_SAVE);
  ciir_as_fir_case_d(size, size/4, w, VSIP_STATE_NO_SAVE);
}

int
main(int argc, char** argv)
{
  vsip_init(0);

  test_iir_as_fir_d();
  test_ciir_as_fir_d();
  vsip_finalize(0);
  return 0;
}

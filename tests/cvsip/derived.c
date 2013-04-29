/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    tests/cvsip/derived.c
    @author  Stefan Seefeld
    @date    2009-02-02
    @brief   derived view tests. See issue248
*/
#include <vsip.h>
#include "output.h"
#include "test.h"

int
main(int argc, char **argv)
{
  size_t i;

  vsip_init(0);

  vsip_scalar_f data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vsip_scalar_f imag[16] = {0};
  vsip_block_f *block = vsip_blockbind_f(data, 16, VSIP_MEM_NONE);
  vsip_blockadmit_f(block, 1);
  vsip_cblock_f *cblock = vsip_cblockbind_f(data, imag, 16, VSIP_MEM_NONE);
  vsip_cblockadmit_f(cblock, 1);
  vsip_mview_f *src = vsip_mbind_f(block, 0, 4, 4, 1, 4);

  /* the reference matrix */
  vsip_cmview_f *ref = vsip_cmbind_f(cblock, 0, 4, 4, 1, 4);
  
  /* the test matrix */
  vsip_cmview_f *test = vsip_cmcreate_f(4, 4, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmfill_f(vsip_cmplx_f(0.f, 0.f), test);

  vsip_mview_f *real = vsip_mrealview_f(test);

  for(i = 0; i < 4; ++i)
  {
    vsip_vview_f *in = vsip_mrowview_f(src, i);
    vsip_vview_f *out = vsip_mrowview_f(real, i);

    vsip_vcopy_f_f(in, out);

    vsip_vdestroy_f(in);
    vsip_vdestroy_f(out);
  }

  test_assert(cmequal_f(test, ref));

  vsip_cmalldestroy_f(test);
  vsip_cmalldestroy_f(ref);
  vsip_malldestroy_f(src);

  vsip_finalize(0);
  return 0;
};

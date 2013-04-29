/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    tests/cvsip/matrix.c
    @author  Stefan Seefeld
    @date    2008-05-02
    @brief   matrix tests.
*/
#include <vsip.h>
#include "output.h"
#include "test.h"

#define DEBUG 1

int
main(int argc, char **argv)
{
  size_t i, j;

  vsip_init(0);

  vsip_scalar_f data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vsip_block_f *block = vsip_blockbind_f(data, 16, VSIP_MEM_NONE);
  vsip_blockadmit_f(block, 1);
  vsip_mview_f *src = vsip_mbind_f(block, 0, 4, 4, 1, 4);

  vsip_mview_f *dest1 = vsip_mcreate_f(4, 4, VSIP_ROW, VSIP_MEM_NONE);
  vsip_mview_f *dest2 = vsip_mcreate_f(4, 4, VSIP_COL, VSIP_MEM_NONE);
  vsip_mview_f *r = vsip_mcreate_f(4, 4, VSIP_ROW, VSIP_MEM_NONE);

  vsip_mview_f *trans = vsip_mtransview_f(src);
  vsip_vview_f *row = vsip_mrowview_f(trans, 2);
  vsip_vview_f *col = vsip_mcolview_f(trans, 3);

  vsip_mcopy_f_f(src, dest1);
  test_assert(mequal_f(src, dest1));

  vsip_mcopy_f_f(src, dest2);
  test_assert(mequal_f(src, dest2));

  vsip_msub_f(dest1, dest2, r);
  moutput_f(r);

#if DEBUG
  printf("source:\n");
  moutput_f(src);
  printf("\n");

  printf("trans:\n");
  moutput_f(trans);
  printf("\n");

  printf("row 2:\n");
  voutput_f(row);
  printf("\n");

  printf("col 3:\n");
  voutput_f(col);
  printf("\n");

  printf("dest1:\n");
  moutput_f(dest1);
  printf("\n");

  printf("dest2:\n");
  moutput_f(dest2);
  printf("\n");
#endif

  vsip_malldestroy_f(dest2);
  vsip_malldestroy_f(dest1);
  vsip_malldestroy_f(src);

  vsip_finalize(0);
  return 0;
};

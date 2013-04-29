/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    tests/cvsip/cmatrix.c
    @author  Stefan Seefeld
    @date    2008-05-02
    @brief   complex matrix tests.
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

  vsip_scalar_f real_inter[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vsip_scalar_f real_data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vsip_scalar_f imag_data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vsip_cblock_f *block = vsip_cblockbind_f(real_data, imag_data, 16, VSIP_MEM_NONE);
  vsip_cblockadmit_f(block, 1);
  vsip_cmview_f *src = vsip_cmbind_f(block, 0, 4, 4, 1, 4);
  
  vsip_cmview_f *dest1 = vsip_cmcreate_f(4, 4, VSIP_ROW, VSIP_MEM_NONE);
  vsip_cmview_f *dest2 = vsip_cmcreate_f(4, 4, VSIP_COL, VSIP_MEM_NONE);

  vsip_cmview_f *trans = vsip_cmtransview_f(src);
  vsip_cvview_f *row = vsip_cmrowview_f(trans, 2);
  vsip_cvview_f *col = vsip_cmcolview_f(trans, 3);

  vsip_cmcopy_f_f(src, dest1);
  test_assert(cmequal_f(src, dest1));

  vsip_cmcopy_f_f(src, dest2);
  test_assert(cmequal_f(src, dest2));

#if DEBUG
  printf("source:\n");
  cmoutput_f(src);
  printf("\n");

  printf("trans:\n");
  cmoutput_f(trans);
  printf("\n");

  printf("row 2:\n");
  cvoutput_f(row);
  printf("\n");

  printf("col 3:\n");
  cvoutput_f(col);
  printf("\n");

  printf("dest1:\n");
  cmoutput_f(dest1);
  printf("\n");

  printf("dest2:\n");
  cmoutput_f(dest2);
  printf("\n");
#endif

  vsip_cmalldestroy_f(dest2);
  vsip_cmalldestroy_f(dest1);
  vsip_cmalldestroy_f(src);

  vsip_finalize(0);
  return 0;
};

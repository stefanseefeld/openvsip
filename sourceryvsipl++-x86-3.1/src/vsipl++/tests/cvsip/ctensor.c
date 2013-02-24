/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    tests/cvsip/tensor.c
    @author  Stefan Seefeld
    @date    2008-05-02
    @brief   tensor tests.
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

  vsip_scalar_f data[27] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
  vsip_cblock_f *block = vsip_cblockbind_f(data, data, 27, VSIP_MEM_NONE);
  vsip_cblockadmit_f(block, 1);
  vsip_ctview_f *src = vsip_ctbind_f(block, 0, 9, 3, 3, 3, 1, 3);

  vsip_ctview_f *dest1 = vsip_ctcreate_f(3, 3, 3, VSIP_TRAILING, VSIP_MEM_NONE);
  vsip_ctview_f *dest2 = vsip_ctcreate_f(3, 3, 3, VSIP_LEADING, VSIP_MEM_NONE);

  vsip_tview_f *real = vsip_trealview_f(src);
  vsip_tview_f *imag = vsip_timagview_f(src);

  vsip_cmview_f *plane_yx = vsip_ctmatrixview_f(src, VSIP_TMYX, 1);
  vsip_cmview_f *plane_zx = vsip_ctmatrixview_f(src, VSIP_TMZX, 1);
  vsip_cmview_f *plane_zy = vsip_ctmatrixview_f(src, VSIP_TMZY, 1);

  vsip_ctcopy_f_f(src, dest1);
  test_assert(ctequal_f(src, dest1));

  vsip_ctcopy_f_f(src, dest2);
  test_assert(ctequal_f(src, dest2));

#if DEBUG
  printf("source:\n");
  ctoutput_f(src);
  printf("\n");

  printf("dest1:\n");
  ctoutput_f(dest1);
  printf("\n");

  printf("dest2:\n");
  ctoutput_f(dest2);
  printf("\n");

  printf("yx plane:\n");
  cmoutput_f(plane_yx);
  printf("\n");

  printf("zx plane:\n");
  cmoutput_f(plane_zx);
  printf("\n");

  printf("zy plane:\n");
  cmoutput_f(plane_zy);
  printf("\n");

#endif

  vsip_cmdestroy_f(plane_yx);
  vsip_cmdestroy_f(plane_zx);
  vsip_cmdestroy_f(plane_zy);
  vsip_ctalldestroy_f(dest2);
  vsip_ctalldestroy_f(dest1);
  vsip_ctalldestroy_f(src);

  vsip_finalize(0);
  return 0;
};

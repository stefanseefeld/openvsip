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
  vsip_block_f *block = vsip_blockbind_f(data, 27, VSIP_MEM_NONE);
  vsip_blockadmit_f(block, 1);
  vsip_tview_f *src = vsip_tbind_f(block, 0, 9, 3, 3, 3, 1, 3);

  vsip_tview_f *dest1 = vsip_tcreate_f(3, 3, 3, VSIP_TRAILING, VSIP_MEM_NONE);
  vsip_tview_f *dest2 = vsip_tcreate_f(3, 3, 3, VSIP_LEADING, VSIP_MEM_NONE);

  vsip_mview_f *plane_yx = vsip_tmatrixview_f(src, VSIP_TMYX, 1);
  vsip_mview_f *plane_zx = vsip_tmatrixview_f(src, VSIP_TMZX, 1);
  vsip_mview_f *plane_zy = vsip_tmatrixview_f(src, VSIP_TMZY, 1);

  vsip_tcopy_f_f(src, dest1);
  test_assert(tequal_f(src, dest1));

  vsip_tcopy_f_f(src, dest2);
  test_assert(tequal_f(src, dest2));

#if DEBUG
  printf("source:\n");
  toutput_f(src);
  printf("\n");

  printf("dest1:\n");
  toutput_f(dest1);
  printf("\n");

  printf("dest2:\n");
  toutput_f(dest2);
  printf("\n");

  printf("yx plane:\n");
  moutput_f(plane_yx);
  printf("\n");

  printf("zx plane:\n");
  moutput_f(plane_zx);
  printf("\n");

  printf("zy plane:\n");
  moutput_f(plane_zy);
  printf("\n");

#endif

  vsip_mdestroy_f(plane_yx);
  vsip_mdestroy_f(plane_zx);
  vsip_mdestroy_f(plane_zy);
  vsip_talldestroy_f(dest2);
  vsip_talldestroy_f(dest1);
  vsip_talldestroy_f(src);

  vsip_finalize(0);
  return 0;
};

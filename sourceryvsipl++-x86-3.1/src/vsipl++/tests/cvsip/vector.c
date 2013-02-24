/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    tests/cvsip/vector.c
    @author  Stefan Seefeld
    @date    2008-05-02
    @brief   vector tests.
*/
#include <vsip.h>
#include "output.h"
#include "test.h"

#define DEBUG 1

int
main()
{
  vsip_init(0);

  vsip_scalar_f data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vsip_block_f *block = vsip_blockbind_f(data, 16, VSIP_MEM_NONE);
  vsip_blockadmit_f(block, 1);
  vsip_vview_f *src = vsip_vbind_f(block, 1, 2, 5);

  vsip_scalar_f sub_data[16] = {3, 5, 7};
  vsip_block_f *sub_block = vsip_blockbind_f(sub_data, 3, VSIP_MEM_NONE);
  vsip_blockadmit_f(sub_block, 1);
  vsip_vview_f *sub_view_ref = vsip_vbind_f(sub_block, 0, 1, 3);

  vsip_vview_f *dest = vsip_vcreate_f(5, VSIP_MEM_NONE);
  vsip_vview_f *sub = vsip_vsubview_f(dest, 1, 3);

  vsip_vcopy_f_f(src, dest);

  test_assert(vequal_f(src, dest));
  test_assert(vequal_f(sub, sub_view_ref));

#if DEBUG
  printf("destination:\n");
  voutput_f(dest);
  printf("\n");
  printf("subview:\n");
  voutput_f(sub);
  printf("\n");
  voutput_f(sub_view_ref);
  printf("\n");
#endif
  vsip_vdestroy_f(sub);
  vsip_valldestroy_f(src);
  vsip_valldestroy_f(dest);
  vsip_finalize(0);
  return 0;
};

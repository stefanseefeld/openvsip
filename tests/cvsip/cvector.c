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

int
main(int argc, char **argv)
{
  int i;

  vsip_init(0);

  vsip_scalar_f real_inter[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vsip_scalar_f real_data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vsip_scalar_f imag_data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vsip_cblock_f *block = vsip_cblockbind_f(real_data, imag_data, 16, VSIP_MEM_NONE);
  vsip_cvview_f *src = vsip_cvbind_f(block, 1, 2, 5);

  vsip_cblockadmit_f(block, 1);
  vsip_cvview_f *dest = vsip_cvcreate_f(5, VSIP_MEM_NONE);
  vsip_cvview_f *sub = vsip_cvsubview_f(dest, 1, 3);
  vsip_vview_f *real = vsip_vrealview_f(sub);
  vsip_vview_f *imag = vsip_vimagview_f(sub);

  vsip_cvcopy_f_f(src, dest);

  test_assert(cvequal_f(src, dest));

#if DEBUG
  printf("destination:\n");
  cvoutput_f(dest);
  printf("\n");
  printf("subview:\n");
  cvoutput_f(sub);
  printf("\n");
  printf("realview / imagview:\n");
  for (i = 0; i != vsip_vgetlength_f(real); ++i)
    printf("(%f %f) ", vsip_vget_f(real, i), vsip_vget_f(imag, i));
  printf("\n");
#endif

  vsip_vdestroy_f(imag);
  vsip_vdestroy_f(real);
  vsip_cvdestroy_f(sub);
  vsip_cvalldestroy_f(dest);
  vsip_cvalldestroy_f(src);
  vsip_finalize(0);
  return 0;
};

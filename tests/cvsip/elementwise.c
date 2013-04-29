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

  vsip_scalar_f data[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  vsip_block_f *block = vsip_blockbind_f(data, 16, VSIP_MEM_NONE);
  vsip_blockadmit_f(block, 1);
  vsip_vview_f *v = vsip_vbind_f(block, 0, 1, 16);
  vsip_vview_f *a = vsip_vcreate_f(16, VSIP_MEM_NONE);
  vsip_vcopy_f_f(v, a);
  vsip_vview_f *b = vsip_vcreate_f(16, VSIP_MEM_NONE);
  vsip_vcopy_f_f(v, b);
  vsip_vview_f *r = vsip_vcreate_f(16, VSIP_MEM_NONE);
  vsip_vadd_f(a, b, r);

#if DEBUG
  printf("a:\n");
  voutput_f(a);
  printf("\n");
  printf("b:\n");
  voutput_f(b);
  printf("\n");
  printf("a+b:\n");
  voutput_f(r);
  printf("\n");
#endif
  vsip_valldestroy_f(r);
  vsip_vdestroy_f(b);
  vsip_valldestroy_f(a);
  vsip_finalize(0);
  return 0;
};

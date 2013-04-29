/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    tests/cvsip/fft.c
    @author  Stefan Seefeld
    @date    2008-05-23
    @brief   test fft operations.
*/
#include <vsip.h>
#include "output.h"
#include "test.h"

#define DEBUG 1

int
main()
{
  vsip_init(0);

  vsip_randstate *rng = vsip_randcreate(0, 1, 1, VSIP_PRNG);
  vsip_fft_d *fft_fwd = vsip_rcfftop_create_d(128, 1, 1, 0);
  // FIXME: the SV++ Fft object should expect 65 as size here !
  vsip_fft_d *fft_inv = vsip_crfftop_create_d(128, 1/128.f, 1, 0);
  vsip_vview_d *a = vsip_vcreate_d(128, VSIP_MEM_NONE);
  vsip_vview_d *c = vsip_vcreate_d(128, VSIP_MEM_NONE);
  vsip_vrandu_d(rng, a);
  vsip_cvview_d *b = vsip_cvcreate_d(65, VSIP_MEM_NONE);
  vsip_rcfftop_d(fft_fwd, a, b);
  vsip_crfftop_d(fft_inv, b, c);
#if DEBUG
  printf("a:\n");
  voutput_d(a);
  printf("\n");
/*   printf("b:\n"); */
/*   cvoutput_d(b); */
/*   printf("\n"); */
  printf("c:\n");
  voutput_d(c);
  printf("\n");
#endif
  test_assert(vequal_d(a, c));
  vsip_valldestroy_d(c);
  vsip_cvalldestroy_d(b);
  vsip_valldestroy_d(a);
  vsip_randdestroy(rng);
  vsip_fft_destroy_d(fft_inv);
  vsip_fft_destroy_d(fft_fwd);
  vsip_finalize(0);
  return EXIT_SUCCESS;
};

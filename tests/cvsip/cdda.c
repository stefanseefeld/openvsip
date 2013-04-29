/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    tests/cvsip/cdda.c
    @author  Stefan Seefeld
    @date    2008-11-10
    @brief   direct data access tests.
*/
#include <vsip_csl.h>
#include "output.h"
#include "test.h"

#define DEBUG 1

int
main()
{
  vsip_init(0);

  vsip_cvview_f *src = vsip_cvcreate_f(8, VSIP_MEM_NONE);
  vsip_cvview_f *ref = vsip_cvcreate_f(8, VSIP_MEM_NONE);
  vsip_cvfill_f(vsip_cmplx_f(0, 0), src);
  vsip_cvview_f *sub = vsip_cvsubview_f(src, 0, 4);
  vsip_csl_cvattr_f cdda;
  vsip_vview_f *real = vsip_vrealview_f(src);
  vsip_vview_f *imag = vsip_vimagview_f(src);
  vsip_csl_vattr_f dda;
  int i;
  /* test a manual dda ramp of a dense vector. */

  vsip_cvramp_f(vsip_cmplx_f(0, 0), vsip_cmplx_f(1, 1), ref);
  vsip_csl_cvgetattrib_f(src, &cdda);
  test_assert(cdda.data_r);
  if (cdda.data_i) /* split */
    for (i = 0; i != cdda.length; ++i)
    {
      cdda.data_r[i * cdda.stride] = i;
      cdda.data_i[i * cdda.stride] = i;
    }
   else
    for (i = 0; i != cdda.length; ++i)
    {
      cdda.data_r[2 * i * cdda.stride] = i;
      cdda.data_r[2 * i * cdda.stride + 1] = i;
    }

  test_assert(cvequal_f(src, ref));

  /* test a manual dda ramp on real subvectors */

  vsip_csl_vgetattrib_f(real, &dda);
  test_assert(dda.data);
  for (i = 0; i != cdda.length; ++i)
    dda.data[i * dda.stride] = 2 * i;

  vsip_csl_vgetattrib_f(imag, &dda);
  test_assert(dda.data);
  for (i = 0; i != dda.length; ++i)
    dda.data[i * dda.stride] = 1 + 2 * i;

  vsip_cvramp_f(vsip_cmplx_f(0, 1), vsip_cmplx_f(2, 2), ref);
  test_assert(cvequal_f(src, ref));

  vsip_vdestroy_f(imag);
  vsip_vdestroy_f(real);
  vsip_cvdestroy_f(sub);
  vsip_cvalldestroy_f(ref);
  vsip_cvalldestroy_f(src);
  vsip_finalize(0);
  return 0;
};

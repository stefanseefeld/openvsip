//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is made available under the GPL.
// See the accompanying file LICENSE.GPL for details.

#include <vsip_csl.h>
#include "output.h"
#include "test.h"

#define DEBUG 1

int
main()
{
  vsip_init(0);

  vsip_vview_f *src = vsip_vcreate_f(8, VSIP_MEM_NONE);
  vsip_vview_f *ref = vsip_vcreate_f(8, VSIP_MEM_NONE);
  vsip_vfill_f(0.f, src);
  vsip_vview_f *sub = vsip_vsubview_f(src, 0, 4);
  vsip_csl_vattr_f dda;
  int i;

  /* test a manual dda ramp of a dense vector. */

  vsip_vramp_f(0, 1, ref);
  vsip_csl_vgetattrib_f(src, &dda);
  test_assert(dda.data);
  for (i = 0; i != dda.length; ++i)
    dda.data[i * dda.stride] = i;

  test_assert(vequal_f(src, ref));

  /* test a manual dda ramp on non-unit-stride subvectors. */

  vsip_vputstride_f(sub, 2); // sub refers to [0, 2, 4, 6]
  vsip_csl_vgetattrib_f(sub, &dda);
  test_assert(dda.data);
  for (i = 0; i != dda.length; ++i)
    dda.data[i * dda.stride] = 2 * i * dda.stride;

  vsip_vputoffset_f(sub, 1);
  vsip_vputstride_f(sub, 2); // sub refers to [1, 3, 5, 7]
  vsip_csl_vgetattrib_f(sub, &dda);
  test_assert(dda.data);
  for (i = 0; i != dda.length; ++i)
    dda.data[i * dda.stride] = 2 * (1 + i * dda.stride);

  vsip_vramp_f(0, 2, ref);
  test_assert(vequal_f(src, ref));

  vsip_vdestroy_f(sub);
  vsip_valldestroy_f(ref);
  vsip_valldestroy_f(src);
  vsip_finalize(0);
  return 0;
};

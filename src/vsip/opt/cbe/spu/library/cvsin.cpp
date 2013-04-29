/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.
   
   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/opt/cbe/spu/vsip.h>
#include "util.hpp"

void cvsin(float *inout, unsigned int length)
{
  vector float *v = (vector float *)inout;
  for (unsigned int i = 0; i != length/4; ++i, v += 2)
  {
    vector float vr = spu_shuffle(*v, *(v+1), shuffle_0246);
    vector float vi = spu_shuffle(*v, *(v+1), shuffle_1357);
    vector float sinvr;
    vector float cosvr;
    sincosf4(vr, &sinvr, &cosvr);
    vector float sinhvi = sinhf4(vi);
    vector float coshvi = coshf4(vi);
    vr = spu_mul(sinvr, coshvi);
    vi = spu_mul(cosvr, sinhvi);
    *v = spu_shuffle(vr, vi, shuffle_0415);
    *(v+1) = spu_shuffle(vr, vi, shuffle_2637);
  }
}

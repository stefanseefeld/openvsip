/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.
   
   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/opt/cbe/spu/vsip.h>
#include "util.hpp"

void zvcos(float *inout, unsigned int length)
{
  vector float *vr = (vector float *)inout;
  vector float *vi = (vector float *)(inout + length);
  for (unsigned int i = 0; i != length/4; ++i, ++vr, ++vi)
  {
    vector float sinvr;
    vector float cosvr;
    sincosf4(*vr, &sinvr, &cosvr);
    vector float sinhvi = sinhf4(*vi);
    vector float coshvi = coshf4(*vi);
    *vr = spu_mul(cosvr, coshvi);
    *vi = negatef4(spu_mul(sinvr, sinhvi));
  }
}

/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.
   
   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/opt/cbe/spu/vsip.h>
#include <simdmath/divf4.h>
#include "util.hpp"

void zvlog10(float *inout, unsigned int length)
{
  vector float *vr = (vector float *)inout;
  vector float *vi = (vector float *)(inout + length);
  vector float half = ((vector float) { 0.5, 0.5, 0.5, 0.5 });
  for (unsigned int i = 0; i != length/4; ++i, ++vr, ++vi)
  {
    vector float tmp = spu_madd(*vr,*vr,spu_mul(*vi,*vi));
    tmp = log10f4(tmp);
    *vi = atan2f4(*vi,*vr);
    *vr = spu_mul(tmp,half);
    *vi = _divf4(*vi,((vector float) { SM_LN10, SM_LN10, SM_LN10, SM_LN10 }));
  }
}

/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.
   
   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/opt/cbe/spu/vsip.h>
#include <simdmath/divf4.h>
#include "util.hpp"

void cvlog10(float *inout, unsigned int length)
{
  vector float *v = (vector float *)inout;
  vector float half = ((vector float) { 0.5, 0.5, 0.5, 0.5 });
  for (unsigned int i = 0; i != length/4; ++i, v += 2)
  {
    vector float vr = spu_shuffle(*v, *(v+1), shuffle_0246);
    vector float vi = spu_shuffle(*v, *(v+1), shuffle_1357);
    vector float tmp = spu_madd(vr,vr,spu_mul(vi,vi));
    tmp = log10f4(tmp);
    vi = atan2f4(vi,vr);
    vr = spu_mul(tmp,half);
    vi = _divf4(vi,((vector float) { SM_LN10, SM_LN10, SM_LN10, SM_LN10 }));
    *v = spu_shuffle(vr, vi, shuffle_0415);
    *(v+1) = spu_shuffle(vr, vi, shuffle_2637);
  }
}

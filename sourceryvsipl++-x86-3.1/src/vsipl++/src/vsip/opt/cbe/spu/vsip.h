/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.
   
   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef SPU_VSIP_H
#define SPU_VSIP_H

#ifndef __SPU__
#error This header is only for use on the SPU.
#endif

#include <spu_intrinsics.h>
#include <simdmath.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

void zvsin(float *inout, unsigned int);
void cvsin(float *inout, unsigned int);
void zvcos(float *inout, unsigned int);
void cvcos(float *inout, unsigned int);
void zvlog(float *inout, unsigned int);
void cvlog(float *inout, unsigned int);
void zvlog10(float *inout, unsigned int);
void cvlog10(float *inout, unsigned int);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* SPU_VSIP_H */

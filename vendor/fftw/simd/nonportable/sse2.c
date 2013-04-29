/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"
#include "simd.h"

#if HAVE_SSE2

const union uvec X(sse2_pm) = {
     { 0x00000000, 0x00000000, 0x00000000, 0x80000000 }
};

/* paranoia because of past compiler bugs */
void X(check_alignment_of_sse2_pm)(void)
{
     CK(ALIGNED(&X(sse2_pm)));
}
#endif


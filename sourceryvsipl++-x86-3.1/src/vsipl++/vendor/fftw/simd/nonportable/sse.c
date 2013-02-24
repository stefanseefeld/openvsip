/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"
#include "simd.h"

#if HAVE_SSE

const union uvec X(sse_pmpm) = {
     { 0x00000000, 0x80000000, 0x00000000, 0x80000000 }
};

/* paranoia because of past compiler bugs */
void X(check_alignment_of_sse_pmpm)(void)
{
     CK(ALIGNED(&X(sse_pmpm)));
}

#endif

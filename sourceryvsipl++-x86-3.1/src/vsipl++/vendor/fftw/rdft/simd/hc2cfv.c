/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-rdft.h"
#include "hc2cfv.h"

#if HAVE_SIMD
static int okp(const R *Rp, const R *Ip, const R *Rm, const R *Im, 
	       INT rs, INT mb, INT me, INT ms, 
	       const planner *plnr)
{
     return (RIGHT_CPU()
	     && !NO_SIMDP(plnr)
	     && SIMD_STRIDE_OK(rs)
	     && SIMD_VSTRIDE_OK(ms)
             && ((me - mb) % VL) == 0
             && ((mb - 1) % VL) == 0 /* twiddle factors alignment */
	     && ALIGNED(Rp)
	     && ALIGNED(Rm)
	     && Ip == Rp + 1
	     && Im == Rm + 1);
}

const hc2c_genus GENUS = { okp, R2HC, VL };
#endif

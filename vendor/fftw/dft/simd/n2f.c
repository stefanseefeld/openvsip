/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-dft.h"
#include "n2f.h"

#if HAVE_SIMD
static int okp(const kdft_desc *d,
               const R *ri, const R *ii, const R *ro, const R *io,
               INT is, INT os, INT vl, INT ivs, INT ovs, 
	       const planner *plnr)
{
     return (RIGHT_CPU()
             && ALIGNEDA(ri)
             && ALIGNEDA(ro)
	     && !NO_SIMDP(plnr)
	     && SIMD_STRIDE_OKA(is)
	     && SIMD_VSTRIDE_OKA(ivs)
	     && SIMD_VSTRIDE_OKA(os) /* os == 2 enforced by codelet */
	     && SIMD_STRIDE_OKPAIR(ovs)
             && ii == ri + 1
             && io == ro + 1
             && (vl % VL) == 0
             && (!d->is || (d->is == is))
             && (!d->os || (d->os == os))
             && (!d->ivs || (d->ivs == ivs))
             && (!d->ovs || (d->ovs == ovs))
          );
}

const kdft_genus GENUS = { okp, VL };
#endif

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-dft.h"
#include "n1b.h"

#if HAVE_SIMD
static int okp(const kdft_desc *d,
               const R *ri, const R *ii, const R *ro, const R *io,
               INT is, INT os, INT vl, INT ivs, INT ovs, 
	       const planner *plnr)
{
     return (RIGHT_CPU()
             && ALIGNED(ii)
             && ALIGNED(io)
	     && !NO_SIMDP(plnr)
	     && SIMD_STRIDE_OK(is)
	     && SIMD_STRIDE_OK(os)
	     && SIMD_VSTRIDE_OK(ivs)
	     && SIMD_VSTRIDE_OK(ovs)
             && ri == ii + 1
             && ro == io + 1
             && (vl % VL) == 0
             && (!d->is || (d->is == is))
             && (!d->os || (d->os == os))
             && (!d->ivs || (d->ivs == ivs))
             && (!d->ovs || (d->ovs == ovs))
          );
}

const kdft_genus GENUS = { okp, VL };
#endif

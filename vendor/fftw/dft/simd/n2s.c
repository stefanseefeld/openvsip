/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-dft.h"

#if HAVE_SIMD
#include "n2s.h"

static int okp(const kdft_desc *d,
               const R *ri, const R *ii, const R *ro, const R *io,
               INT is, INT os, INT vl, INT ivs, INT ovs, 
	       const planner *plnr)
{
     return (RIGHT_CPU()
	     && !NO_SIMDP(plnr)
             && ALIGNEDA(ri)
             && ALIGNEDA(ii)
             && ALIGNEDA(ro)
             && ALIGNEDA(io)
	     && SIMD_STRIDE_OKA(is)
	     && ivs == 1
	     && os == 1
	     && SIMD_STRIDE_OKA(ovs)
             && (vl % (2 * VL)) == 0
             && (!d->is || (d->is == is))
             && (!d->os || (d->os == os))
             && (!d->ivs || (d->ivs == ivs))
             && (!d->ovs || (d->ovs == ovs))
          );
}

const kdft_genus GENUS = { okp, 2 * VL };
#endif

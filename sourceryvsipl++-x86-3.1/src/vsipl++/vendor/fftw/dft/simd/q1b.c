/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-dft.h"
#include "q1b.h"

#if HAVE_SIMD
static int okp(const ct_desc *d,
	       const R *rio, const R *iio, 
	       INT rs, INT vs, INT m, INT mb, INT me, INT ms,
	       const planner *plnr)
{
     return (RIGHT_CPU()
             && ALIGNED(iio)
	     && !NO_SIMDP(plnr)
	     && SIMD_STRIDE_OK(rs)
	     && SIMD_STRIDE_OK(vs)
	     && SIMD_VSTRIDE_OK(ms)
	     && rio == iio + 1
             && (m % VL) == 0
             && (mb % VL) == 0
             && (me % VL) == 0
	     && (!d->rs || (d->rs == rs))
	     && (!d->vs || (d->vs == vs))
	     && (!d->ms || (d->ms == ms))
	  );
}
const ct_genus GENUS = { okp, VL };
#endif

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-dft.h"

#if HAVE_SIMD
#include "ts.h"

static int okp(const ct_desc *d,
	       const R *rio, const R *iio, 
	       INT rs, INT vs, INT m, INT mb, INT me, INT ms,
	       const planner *plnr)
{
     UNUSED(rio);
     UNUSED(iio);
     return (RIGHT_CPU()
	     && !NO_SIMDP(plnr)
	     && ALIGNEDA(rio)
	     && ALIGNEDA(iio)
	     && SIMD_STRIDE_OKA(rs)
	     && ms == 1
             && (m % (2 * VL)) == 0
             && (mb % (2 * VL)) == 0
             && (me % (2 * VL)) == 0
	     && (!d->rs || (d->rs == rs))
	     && (!d->vs || (d->vs == vs))
	     && (!d->ms || (d->ms == ms))
	  );
}

const ct_genus GENUS = { okp, 2 * VL };

#endif

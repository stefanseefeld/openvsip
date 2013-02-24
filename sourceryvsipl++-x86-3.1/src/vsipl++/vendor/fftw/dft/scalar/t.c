/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-dft.h"
#include "t.h"

static int okp(const ct_desc *d,
	       const R *rio, const R *iio, 
	       INT rs, INT vs, INT m, INT mb, INT me, INT ms,
	       const planner *plnr)
{
     UNUSED(rio); UNUSED(iio); UNUSED(m); UNUSED(mb); UNUSED(me); UNUSED(plnr);
     return (1
	     && (!d->rs || (d->rs == rs))
	     && (!d->vs || (d->vs == vs))
	     && (!d->ms || (d->ms == ms))
	  );
}

const ct_genus GENUS = { okp, 1 };

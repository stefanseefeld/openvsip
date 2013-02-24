/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-dft.h"
#include "n.h"

static int okp(const kdft_desc *d,
	       const R *ri, const R *ii, 
	       const R *ro, const R *io,
	       INT is, INT os, INT vl, INT ivs, INT ovs,
	       const planner *plnr)
{
     UNUSED(ri); UNUSED(ii); UNUSED(ro); UNUSED(io); UNUSED(vl); UNUSED(plnr);
     return (1
	     && (!d->is || (d->is == is))
	     && (!d->os || (d->os == os))
	     && (!d->ivs || (d->ivs == ivs))
	     && (!d->ovs || (d->ovs == ovs))
	  );
}

const kdft_genus GENUS = { okp, 1 };

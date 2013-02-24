/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-rdft.h"
#include "hc2cf.h"

static int okp(const R *Rp, const R *Ip, const R *Rm, const R *Im, 
	       INT rs, INT mb, INT me, INT ms, 
	       const planner *plnr)
{
     UNUSED(Rp); UNUSED(Ip); UNUSED(Rm); UNUSED(Im);
     UNUSED(rs); UNUSED(mb); UNUSED(me); UNUSED(ms); UNUSED(plnr);

     return 1;
}

const hc2c_genus GENUS = { okp, R2HC, 1 };

#undef GENUS
#include "hc2cb.h"

const hc2c_genus GENUS = { okp, HC2R, 1 };

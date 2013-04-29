/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "rdft.h"

rdft_kind *X(map_r2r_kind)(int rank, const X(r2r_kind) * kind);

X(plan) XGURU(r2r)(int rank, const IODIM *dims,
			 int howmany_rank,
			 const IODIM *howmany_dims,
			 R *in, R *out,
			 const X(r2r_kind) * kind, unsigned flags)
{
     X(plan) p;
     rdft_kind *k;

     if (!GURU_KOSHERP(rank, dims, howmany_rank, howmany_dims)) return 0;

     k = X(map_r2r_kind)(rank, kind);
     p = X(mkapiplan)(
	  0, flags,
	  X(mkproblem_rdft_d)(MKTENSOR_IODIMS(rank, dims, 1, 1),
			      MKTENSOR_IODIMS(howmany_rank, howmany_dims,
						 1, 1), 
			      TAINT_UNALIGNED(in, flags),
			      TAINT_UNALIGNED(out, flags), k));
     X(ifree0)(k);
     return p;
}

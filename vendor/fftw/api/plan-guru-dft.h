/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "dft.h"

X(plan) XGURU(dft)(int rank, const IODIM *dims,
			 int howmany_rank, const IODIM *howmany_dims,
			 C *in, C *out, int sign, unsigned flags)
{
     R *ri, *ii, *ro, *io;

     if (!GURU_KOSHERP(rank, dims, howmany_rank, howmany_dims)) return 0;

     EXTRACT_REIM(sign, in, &ri, &ii);
     EXTRACT_REIM(sign, out, &ro, &io);

     return X(mkapiplan)(
	  sign, flags,
	  X(mkproblem_dft_d)(MKTENSOR_IODIMS(rank, dims, 2, 2),
			     MKTENSOR_IODIMS(howmany_rank, howmany_dims,
						2, 2),
			     TAINT_UNALIGNED(ri, flags),
			     TAINT_UNALIGNED(ii, flags), 
			     TAINT_UNALIGNED(ro, flags),
			     TAINT_UNALIGNED(io, flags)));
}

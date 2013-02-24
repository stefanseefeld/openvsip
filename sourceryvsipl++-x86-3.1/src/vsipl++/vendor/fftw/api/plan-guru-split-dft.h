/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "dft.h"

X(plan) XGURU(split_dft)(int rank, const IODIM *dims,
			       int howmany_rank, const IODIM *howmany_dims,
			       R *ri, R *ii, R *ro, R *io, unsigned flags)
{
     if (!GURU_KOSHERP(rank, dims, howmany_rank, howmany_dims)) return 0;

     return X(mkapiplan)(
	  ii - ri == 1 && io - ro == 1 ? FFT_SIGN : -FFT_SIGN, flags,
	  X(mkproblem_dft_d)(MKTENSOR_IODIMS(rank, dims, 1, 1),
			     MKTENSOR_IODIMS(howmany_rank, howmany_dims,
						1, 1),
			     TAINT_UNALIGNED(ri, flags),
			     TAINT_UNALIGNED(ii, flags), 
			     TAINT_UNALIGNED(ro, flags),
			     TAINT_UNALIGNED(io, flags)));
}

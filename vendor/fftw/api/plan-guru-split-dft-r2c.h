/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "rdft.h"

X(plan) XGURU(split_dft_r2c)(int rank, const IODIM *dims,
			     int howmany_rank,
			     const IODIM *howmany_dims,
			     R *in, R *ro, R *io, unsigned flags)
{
     if (!GURU_KOSHERP(rank, dims, howmany_rank, howmany_dims)) return 0;

     return X(mkapiplan)(
	  0, flags,
	  X(mkproblem_rdft2_d_3pointers)(
	       MKTENSOR_IODIMS(rank, dims, 1, 1),
	       MKTENSOR_IODIMS(howmany_rank, howmany_dims, 1, 1),
	       TAINT_UNALIGNED(in, flags),
	       TAINT_UNALIGNED(ro, flags),
	       TAINT_UNALIGNED(io, flags), R2HC));
}

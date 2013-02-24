/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "rdft.h"

X(plan) XGURU(split_dft_c2r)(int rank, const IODIM *dims,
			     int howmany_rank, const IODIM *howmany_dims,
			     R *ri, R *ii, R *out, unsigned flags)
{
     if (!GURU_KOSHERP(rank, dims, howmany_rank, howmany_dims)) return 0;

     if (out != ri)
	  flags |= FFTW_DESTROY_INPUT;
     return X(mkapiplan)(
	  0, flags, 
	  X(mkproblem_rdft2_d_3pointers)(
	       MKTENSOR_IODIMS(rank, dims, 1, 1),
	       MKTENSOR_IODIMS(howmany_rank, howmany_dims, 1, 1),
	       TAINT_UNALIGNED(out, flags),
	       TAINT_UNALIGNED(ri, flags),
	       TAINT_UNALIGNED(ii, flags), HC2R));
}

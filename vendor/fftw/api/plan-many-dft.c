/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "dft.h"

#define N0(nembed)((nembed) ? (nembed) : n)

X(plan) X(plan_many_dft)(int rank, const int *n,
			 int howmany,
			 C *in, const int *inembed,
			 int istride, int idist,
			 C *out, const int *onembed,
			 int ostride, int odist, int sign, unsigned flags)
{
     R *ri, *ii, *ro, *io;

     if (!X(many_kosherp)(rank, n, howmany)) return 0;

     EXTRACT_REIM(sign, in, &ri, &ii);
     EXTRACT_REIM(sign, out, &ro, &io);

     return 
	  X(mkapiplan)(sign, flags,
		       X(mkproblem_dft_d)(
			    X(mktensor_rowmajor)(rank, n, 
						 N0(inembed), N0(onembed),
						 2 * istride, 2 * ostride),
			    X(mktensor_1d)(howmany, 2 * idist, 2 * odist),
			    TAINT_UNALIGNED(ri, flags),
			    TAINT_UNALIGNED(ii, flags),
			    TAINT_UNALIGNED(ro, flags),
			    TAINT_UNALIGNED(io, flags)));
}

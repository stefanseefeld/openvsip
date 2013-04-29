/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "rdft.h"

X(plan) X(plan_many_dft_r2c)(int rank, const int *n,
			     int howmany,
			     R *in, const int *inembed,
			     int istride, int idist,
			     C *out, const int *onembed,
			     int ostride, int odist, unsigned flags)
{
     R *ro, *io;
     int *nfi, *nfo;
     int inplace;
     X(plan) p;

     if (!X(many_kosherp)(rank, n, howmany)) return 0;

     EXTRACT_REIM(FFT_SIGN, out, &ro, &io);
     inplace = in == ro;

     p = X(mkapiplan)(
	  0, flags, 
	  X(mkproblem_rdft2_d_3pointers)(
	       X(mktensor_rowmajor)(
		    rank, n,
		    X(rdft2_pad)(rank, n, inembed, inplace, 0, &nfi),
		    X(rdft2_pad)(rank, n, onembed, inplace, 1, &nfo),
		    istride, 2 * ostride), 
	       X(mktensor_1d)(howmany, idist, 2 * odist),
	       TAINT_UNALIGNED(in, flags),
	       TAINT_UNALIGNED(ro, flags), TAINT_UNALIGNED(io, flags),
	       R2HC));

     X(ifree0)(nfi);
     X(ifree0)(nfo);
     return p;
}

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "rdft.h"

X(plan) X(plan_many_dft_c2r)(int rank, const int *n,
			     int howmany,
			     C *in, const int *inembed,
			     int istride, int idist,
			     R *out, const int *onembed,
			     int ostride, int odist, unsigned flags)
{
     R *ri, *ii;
     int *nfi, *nfo;
     int inplace;
     X(plan) p;

     if (!X(many_kosherp)(rank, n, howmany)) return 0;

     EXTRACT_REIM(FFT_SIGN, in, &ri, &ii);
     inplace = out == ri;

     if (!inplace)
	  flags |= FFTW_DESTROY_INPUT;
     p = X(mkapiplan)(
	  0, flags,
	  X(mkproblem_rdft2_d_3pointers)(
	       X(mktensor_rowmajor)(
		    rank, n, 
		    X(rdft2_pad)(rank, n, inembed, inplace, 1, &nfi),
		    X(rdft2_pad)(rank, n, onembed, inplace, 0, &nfo),
		    2 * istride, ostride),
	       X(mktensor_1d)(howmany, 2 * idist, odist),
	       TAINT_UNALIGNED(out, flags),
	       TAINT_UNALIGNED(ri, flags), TAINT_UNALIGNED(ii, flags),
	       HC2R));

     X(ifree0)(nfi);
     X(ifree0)(nfo);
     return p;
}

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "rdft.h"

/* Check if the vecsz/sz strides are consistent with the problem
   being in-place for vecsz.dim[vdim], or for all dimensions
   if vdim == RNK_MINFTY.  We can't just use tensor_inplace_strides
   because rdft transforms have the unfortunate property of
   differing input and output sizes.   This routine is not
   exhaustive; we only return 1 for the most common case.  */
int X(rdft2_inplace_strides)(const problem_rdft2 *p, int vdim)
{
     INT N, Nc;
     INT rs, cs;
     int i;
     
     for (i = 0; i + 1 < p->sz->rnk; ++i)
	  if (p->sz->dims[i].is != p->sz->dims[i].os)
	       return 0;

     if (!FINITE_RNK(p->vecsz->rnk) || p->vecsz->rnk == 0)
	  return 1;
     if (!FINITE_RNK(vdim)) { /* check all vector dimensions */
	  for (vdim = 0; vdim < p->vecsz->rnk; ++vdim)
	       if (!X(rdft2_inplace_strides)(p, vdim))
		    return 0;
	  return 1;
     }

     A(vdim < p->vecsz->rnk);
     if (p->sz->rnk == 0)
	  return(p->vecsz->dims[vdim].is == p->vecsz->dims[vdim].os);

     N = X(tensor_sz)(p->sz);
     Nc = (N / p->sz->dims[p->sz->rnk-1].n) *
	  (p->sz->dims[p->sz->rnk-1].n/2 + 1);
     X(rdft2_strides)(p->kind, p->sz->dims + p->sz->rnk - 1, &rs, &cs);

     /* the factor of 2 comes from the fact that RS is the stride
	of p->r0 and p->r1, which is twice as large as the strides
	in the r2r case */
     return(p->vecsz->dims[vdim].is == p->vecsz->dims[vdim].os
	    && (X(iabs)(2 * p->vecsz->dims[vdim].os)
		>= X(imax)(2 * Nc * X(iabs)(cs), N * X(iabs)(rs))));
}

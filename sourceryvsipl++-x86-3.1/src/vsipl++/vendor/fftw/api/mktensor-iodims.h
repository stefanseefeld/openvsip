/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

tensor *MKTENSOR_IODIMS(int rank, const IODIM *dims, int is, int os)
{
     int i;
     tensor *x = X(mktensor)(rank);

     if (FINITE_RNK(rank)) {
          for (i = 0; i < rank; ++i) {
               x->dims[i].n = dims[i].n;
               x->dims[i].is = dims[i].is * is;
               x->dims[i].os = dims[i].os * os;
          }
     }
     return x;
}

static int iodims_kosherp(int rank, const IODIM *dims, int allow_minfty)
{
     int i;

     if (rank < 0) return 0;

     if (allow_minfty) {
	  if (!FINITE_RNK(rank)) return 1;
	  for (i = 0; i < rank; ++i)
	       if (dims[i].n < 0) return 0;
     } else {
	  if (!FINITE_RNK(rank)) return 0;
	  for (i = 0; i < rank; ++i)
	       if (dims[i].n <= 0) return 0;
     }

     return 1;
}

int GURU_KOSHERP(int rank, const IODIM *dims,
		 int howmany_rank, const IODIM *howmany_dims)
{
     return (iodims_kosherp(rank, dims, 0) &&
	     iodims_kosherp(howmany_rank, howmany_dims, 1));
}

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "dft.h"

/* fill a complex array with zeros. */
static void recur(const iodim *dims, int rnk, R *ri, R *ii)
{
     if (rnk == RNK_MINFTY)
          return;
     else if (rnk == 0)
          ri[0] = ii[0] = K(0.0);
     else if (rnk > 0) {
          INT i, n = dims[0].n;
          INT is = dims[0].is;

	  if (rnk == 1) {
	       /* this case is redundant but faster */
	       for (i = 0; i < n; ++i)
		    ri[i * is] = ii[i * is] = K(0.0);
	  } else {
	       for (i = 0; i < n; ++i)
		    recur(dims + 1, rnk - 1, ri + i * is, ii + i * is);
	  }
     }
}


void X(dft_zerotens)(tensor *sz, R *ri, R *ii)
{
     recur(sz->dims, sz->rnk, ri, ii);
}

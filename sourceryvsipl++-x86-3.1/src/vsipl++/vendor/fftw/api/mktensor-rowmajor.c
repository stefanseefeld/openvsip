/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

tensor *X(mktensor_rowmajor)(int rnk, const int *n,
			     const int *niphys, const int *nophys,
			     int is, int os)
{
     tensor *x = X(mktensor)(rnk);

     if (FINITE_RNK(rnk) && rnk > 0) {
          int i;

          A(n && niphys && nophys);
          x->dims[rnk - 1].is = is;
          x->dims[rnk - 1].os = os;
          x->dims[rnk - 1].n = n[rnk - 1];
          for (i = rnk - 1; i > 0; --i) {
               x->dims[i - 1].is = x->dims[i].is * niphys[i];
               x->dims[i - 1].os = x->dims[i].os * nophys[i];
               x->dims[i - 1].n = n[i - 1];
          }
     }
     return x;
}

static int rowmajor_kosherp(int rnk, const int *n)
{
     int i;

     if (!FINITE_RNK(rnk)) return 0;
     if (rnk < 0) return 0;

     for (i = 0; i < rnk; ++i)
	  if (n[i] <= 0) return 0;

     return 1;
}

int X(many_kosherp)(int rnk, const int *n, int howmany)
{
     return (howmany >= 0) && rowmajor_kosherp(rnk, n);
}

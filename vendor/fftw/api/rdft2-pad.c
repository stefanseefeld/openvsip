/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include <string.h>
#include "api.h"

const int *X(rdft2_pad)(int rnk, const int *n, const int *nembed,
			int inplace, int cmplx, int **nfree)
{
     A(FINITE_RNK(rnk));
     *nfree = 0;
     if (!nembed && rnk > 0) {
          if (inplace || cmplx) {
               int *np = (int *) MALLOC(sizeof(int) * rnk, PROBLEMS);
               memcpy(np, n, sizeof(int) * rnk);
               np[rnk - 1] = (n[rnk - 1] / 2 + 1) * (1 + !cmplx);
               nembed = *nfree = np;
          } else
               nembed = n;
     }
     return nembed;
}

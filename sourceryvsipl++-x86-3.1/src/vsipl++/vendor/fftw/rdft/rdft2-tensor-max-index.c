/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "rdft.h"

/* like X(tensor_max_index), but takes into account the special n/2+1
   final dimension for the complex output/input of an R2HC/HC2R transform. */
INT X(rdft2_tensor_max_index)(const tensor *sz, rdft_kind k)
{
     int i;
     INT n = 0;

     A(FINITE_RNK(sz->rnk));
     for (i = 0; i + 1 < sz->rnk; ++i) {
          const iodim *p = sz->dims + i;
          n += (p->n - 1) * X(imax)(X(iabs)(p->is), X(iabs)(p->os));
     }
     if (i < sz->rnk) {
	  const iodim *p = sz->dims + i;
	  INT is, os;
	  X(rdft2_strides)(k, p, &is, &os);
	  n += X(imax)((p->n - 1) * X(iabs)(is), (p->n/2) * X(iabs)(os));
     }
     return n;
}

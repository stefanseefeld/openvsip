/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "verify.h"

static void recur(int rnk, const bench_iodim *dims0, const bench_iodim *dims1,
		  dotens2_closure *k, 
		  int indx0, int ondx0, int indx1, int ondx1)
{
     if (rnk == 0)
          k->apply(k, indx0, ondx0, indx1, ondx1);
     else {
          int i, n = dims0[0].n;
          int is0 = dims0[0].is;
          int os0 = dims0[0].os;
          int is1 = dims1[0].is;
          int os1 = dims1[0].os;

	  BENCH_ASSERT(n == dims1[0].n);

          for (i = 0; i < n; ++i) {
               recur(rnk - 1, dims0 + 1, dims1 + 1, k,
		     indx0, ondx0, indx1, ondx1);
	       indx0 += is0; ondx0 += os0;
	       indx1 += is1; ondx1 += os1;
	  }
     }
}

void bench_dotens2(const bench_tensor *sz0, const bench_tensor *sz1, dotens2_closure *k)
{
     BENCH_ASSERT(sz0->rnk == sz1->rnk);
     if (sz0->rnk == RNK_MINFTY)
          return;
     recur(sz0->rnk, sz0->dims, sz1->dims, k, 0, 0, 0, 0);
}

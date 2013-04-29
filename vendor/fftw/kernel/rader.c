/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "ifftw.h"

/*
  common routines for Rader solvers 
*/


/* shared twiddle and omega lists, keyed by two/three integers. */
struct rader_tls {
     INT k1, k2, k3;
     R *W;
     int refcnt;
     rader_tl *cdr; 
};

void X(rader_tl_insert)(INT k1, INT k2, INT k3, R *W, rader_tl **tl)
{
     rader_tl *t = (rader_tl *) MALLOC(sizeof(rader_tl), TWIDDLES);
     t->k1 = k1; t->k2 = k2; t->k3 = k3; t->W = W;
     t->refcnt = 1; t->cdr = *tl; *tl = t;
}

R *X(rader_tl_find)(INT k1, INT k2, INT k3, rader_tl *t)
{
     while (t && (t->k1 != k1 || t->k2 != k2 || t->k3 != k3))
	  t = t->cdr;
     if (t) {
	  ++t->refcnt;
	  return t->W;
     } else 
	  return 0;
}

void X(rader_tl_delete)(R *W, rader_tl **tl)
{
     if (W) {
	  rader_tl **tp, *t;

	  for (tp = tl; (t = *tp) && t->W != W; tp = &t->cdr)
	       ;

	  if (t && --t->refcnt <= 0) {
	       *tp = t->cdr;
	       X(ifree)(t->W);
	       X(ifree)(t);
	  }
     }
}

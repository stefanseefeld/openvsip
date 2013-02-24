/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"
#include "simd.h"

#if HAVE_SIMD || HAVE_CELL

R *X(taint)(R *p, INT s)
{
     if (((unsigned)s * sizeof(R)) % ALIGNMENT)
	  p = (R *) (PTRINT(p) | TAINT_BIT);
     if (((unsigned)s * sizeof(R)) % ALIGNMENTA)
	  p = (R *) (PTRINT(p) | TAINT_BITA);
     return p;
}

/* join the taint of two pointers that are supposed to be
   the same modulo the taint */
R *X(join_taint)(R *p1, R *p2)
{
     A(UNTAINT(p1) == UNTAINT(p2));
     return (R *)(PTRINT(p1) | PTRINT(p2));
}
#endif

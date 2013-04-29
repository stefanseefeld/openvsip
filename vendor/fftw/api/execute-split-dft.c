/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "dft.h"

/* guru interface: requires care in alignment, r - i, etcetera. */
void X(execute_split_dft)(const X(plan) p, R *ri, R *ii, R *ro, R *io)
WITH_ALIGNED_STACK({
     plan_dft *pln = (plan_dft *) p->pln;
     pln->apply((plan *) pln, ri, ii, ro, io);
})

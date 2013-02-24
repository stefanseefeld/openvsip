/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "rdft.h"

/* guru interface: requires care in alignment, r - i, etcetera. */
void X(execute_split_dft_r2c)(const X(plan) p, R *in, R *ro, R *io)
WITH_ALIGNED_STACK({
     plan_rdft2 *pln = (plan_rdft2 *) p->pln;
     problem_rdft2 *prb = (problem_rdft2 *) p->prb;
     pln->apply((plan *) pln, in, in + (prb->r1 - prb->r0), ro, io);
})

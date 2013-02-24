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
void X(execute_dft_c2r)(const X(plan) p, C *in, R *out)
WITH_ALIGNED_STACK({
     plan_rdft2 *pln = (plan_rdft2 *) p->pln;
     problem_rdft2 *prb = (problem_rdft2 *) p->prb;
     pln->apply((plan *) pln, out, out + (prb->r1 - prb->r0), in[0], in[0]+1);
})

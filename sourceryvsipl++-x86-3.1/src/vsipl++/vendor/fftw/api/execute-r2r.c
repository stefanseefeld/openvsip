/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "rdft.h"

/* guru interface: requires care in alignment, etcetera. */
void X(execute_r2r)(const X(plan) p, R *in, R *out)
WITH_ALIGNED_STACK({
     plan_rdft *pln = (plan_rdft *) p->pln;
     pln->apply((plan *) pln, in, out);
})

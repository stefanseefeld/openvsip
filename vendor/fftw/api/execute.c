/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

void X(execute)(const X(plan) p)
WITH_ALIGNED_STACK({
     plan *pln = p->pln;
     pln->adt->solve(pln, p->prb);
})

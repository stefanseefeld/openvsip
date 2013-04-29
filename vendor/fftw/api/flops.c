/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

void X(flops)(const X(plan) p, double *add, double *mul, double *fma)
{
     planner *plnr = X(the_planner)();
     opcnt *o = &p->pln->ops;
     *add = o->add; *mul = o->mul; *fma = o->fma;
     if (plnr->cost_hook) {
	  *add = plnr->cost_hook(p->prb, *add, COST_SUM);
	  *mul = plnr->cost_hook(p->prb, *mul, COST_SUM);
	  *fma = plnr->cost_hook(p->prb, *fma, COST_SUM);
     }
}

double X(estimate_cost)(const X(plan) p)
{
     return X(iestimate_cost)(X(the_planner)(), p->pln, p->prb);
}

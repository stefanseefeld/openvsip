/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"

solver *X(mksolver)(size_t size, const solver_adt *adt)
{
     solver *s = (solver *)MALLOC(size, SOLVERS);

     s->adt = adt;
     s->refcnt = 0;
     return s;
}

void X(solver_use)(solver *ego)
{
     ++ego->refcnt;
}

void X(solver_destroy)(solver *ego)
{
     if ((--ego->refcnt) == 0) {
	  if (ego->adt->destroy)
	       ego->adt->destroy(ego);
          X(ifree)(ego);
     }
}

void X(solver_register)(planner *plnr, solver *s)
{
     plnr->adt->register_solver(plnr, s);
}

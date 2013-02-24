/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"

/* constructor */
problem *X(mkproblem)(size_t sz, const problem_adt *adt)
{
     problem *p = (problem *)MALLOC(sz, PROBLEMS);

     p->adt = adt;
     return p;
}

/* destructor */
void X(problem_destroy)(problem *ego)
{
     if (ego)
	  ego->adt->destroy(ego);
}

/* management of unsolvable problems */
static void unsolvable_destroy(problem *ego)
{
     UNUSED(ego);
}

static void unsolvable_hash(const problem *p, md5 *m)
{
     UNUSED(p);
     X(md5puts)(m, "unsolvable");
}

static void unsolvable_print(const problem *ego, printer *p)
{
     UNUSED(ego);
     p->print(p, "(unsolvable)");
}

static void unsolvable_zero(const problem *ego)
{
     UNUSED(ego);
}

static const problem_adt padt =
{
     PROBLEM_UNSOLVABLE,
     unsolvable_hash,
     unsolvable_zero,
     unsolvable_print,
     unsolvable_destroy
};

/* there is no point in malloc'ing this one */
static problem the_unsolvable_problem = { &padt };

problem *X(mkproblem_unsolvable)(void)
{
     return &the_unsolvable_problem;
}

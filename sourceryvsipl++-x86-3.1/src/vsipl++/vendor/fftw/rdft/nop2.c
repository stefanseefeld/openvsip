/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


/* plans for vrank -infty RDFT2s (nothing to do), as well as in-place
   rank-0 HC2R.  Note that in-place rank-0 R2HC is *not* a no-op, because
   we have to set the imaginary parts of the output to zero. */

#include "rdft.h"

static void apply(const plan *ego_, R *r0, R *r1, R *cr, R *ci)
{
     UNUSED(ego_);
     UNUSED(r0);
     UNUSED(r1);
     UNUSED(cr);
     UNUSED(ci);
}

static int applicable(const solver *ego_, const problem *p_)
{
     const problem_rdft2 *p = (const problem_rdft2 *) p_;
     UNUSED(ego_);

     return(0
	    /* case 1 : -infty vector rank */
	    || (p->vecsz->rnk == RNK_MINFTY)
		 
	    /* case 2 : rank-0 in-place rdft, except that
	       R2HC is not a no-op because it sets the imaginary
	       part to 0 */
	    || (1
		&& p->kind != R2HC
		&& p->sz->rnk == 0
		&& FINITE_RNK(p->vecsz->rnk)
		&& (p->r0 == p->cr)
		&& X(rdft2_inplace_strides)(p, RNK_MINFTY)
		 ));
}

static void print(const plan *ego, printer *p)
{
     UNUSED(ego);
     p->print(p, "(rdft2-nop)");
}

static plan *mkplan(const solver *ego, const problem *p, planner *plnr)
{
     static const plan_adt padt = {
	  X(rdft2_solve), X(null_awake), print, X(plan_null_destroy)
     };
     plan_rdft2 *pln;

     UNUSED(plnr);

     if (!applicable(ego, p))
          return (plan *) 0;
     pln = MKPLAN_RDFT2(plan_rdft2, &padt, apply);
     X(ops_zero)(&pln->super.ops);

     return &(pln->super);
}

static solver *mksolver(void)
{
     static const solver_adt sadt = { PROBLEM_RDFT2, mkplan, 0 };
     return MKSOLVER(solver, &sadt);
}

void X(rdft2_nop_register)(planner *p)
{
     REGISTER_SOLVER(p, mksolver());
}

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


/* plans for vrank -infty RDFTs (nothing to do) */

#include "rdft.h"

static void apply(const plan *ego_, R *I, R *O)
{
     UNUSED(ego_);
     UNUSED(I);
     UNUSED(O);
}

static int applicable(const solver *ego_, const problem *p_)
{
     const problem_rdft *p = (const problem_rdft *) p_;
     UNUSED(ego_);
     return 0
	  /* case 1 : -infty vector rank */
	  || (p->vecsz->rnk == RNK_MINFTY)

	  /* case 2 : rank-0 in-place rdft */
	  || (1
	      && p->sz->rnk == 0
	      && FINITE_RNK(p->vecsz->rnk)
	      && p->O == p->I
	      && X(tensor_inplace_strides)(p->vecsz)
	       );
}

static void print(const plan *ego, printer *p)
{
     UNUSED(ego);
     p->print(p, "(rdft-nop)");
}

static plan *mkplan(const solver *ego, const problem *p, planner *plnr)
{
     static const plan_adt padt = {
	  X(rdft_solve), X(null_awake), print, X(plan_null_destroy)
     };
     plan_rdft *pln;

     UNUSED(plnr);

     if (!applicable(ego, p))
          return (plan *) 0;
     pln = MKPLAN_RDFT(plan_rdft, &padt, apply);
     X(ops_zero)(&pln->super.ops);

     return &(pln->super);
}

static solver *mksolver(void)
{
     static const solver_adt sadt = { PROBLEM_RDFT, mkplan, 0 };
     return MKSOLVER(solver, &sadt);
}

void X(rdft_nop_register)(planner *p)
{
     REGISTER_SOLVER(p, mksolver());
}

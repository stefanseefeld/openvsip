/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


/* plans for vrank -infty DFTs (nothing to do) */

#include "dft.h"

static void apply(const plan *ego_, R *ri, R *ii, R *ro, R *io)
{
     UNUSED(ego_);
     UNUSED(ri);
     UNUSED(ii);
     UNUSED(ro);
     UNUSED(io);
}

static int applicable(const solver *ego_, const problem *p_)
{
     const problem_dft *p = (const problem_dft *) p_;

     UNUSED(ego_);

     return 0
	  /* case 1 : -infty vector rank */
	  || (!FINITE_RNK(p->vecsz->rnk))

	  /* case 2 : rank-0 in-place dft */
	  || (1
	      && p->sz->rnk == 0
	      && FINITE_RNK(p->vecsz->rnk)
	      && p->ro == p->ri
	      && X(tensor_inplace_strides)(p->vecsz)
	       );
}

static void print(const plan *ego, printer *p)
{
     UNUSED(ego);
     p->print(p, "(dft-nop)");
}

static plan *mkplan(const solver *ego, const problem *p, planner *plnr)
{
     static const plan_adt padt = {
	  X(dft_solve), X(null_awake), print, X(plan_null_destroy)
     };
     plan_dft *pln;

     UNUSED(plnr);

     if (!applicable(ego, p))
          return (plan *) 0;
     pln = MKPLAN_DFT(plan_dft, &padt, apply);
     X(ops_zero)(&pln->super.ops);

     return &(pln->super);
}

static solver *mksolver(void)
{
     static const solver_adt sadt = { PROBLEM_DFT, mkplan, 0 };
     return MKSOLVER(solver, &sadt);
}

void X(dft_nop_register)(planner *p)
{
     REGISTER_SOLVER(p, mksolver());
}

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "rdft.h"

/* use the apply() operation for RDFT2 problems */
void X(rdft2_solve)(const plan *ego_, const problem *p_)
{
     const plan_rdft2 *ego = (const plan_rdft2 *) ego_;
     const problem_rdft2 *p = (const problem_rdft2 *) p_;
     ego->apply(ego_, 
		UNTAINT(p->r0), UNTAINT(p->r1),
		UNTAINT(p->cr), UNTAINT(p->ci));
}

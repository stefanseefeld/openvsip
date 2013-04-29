/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "rdft.h"

/* use the apply() operation for RDFT problems */
void X(rdft_solve)(const plan *ego_, const problem *p_)
{
     const plan_rdft *ego = (const plan_rdft *) ego_;
     const problem_rdft *p = (const problem_rdft *) p_;
     ego->apply(ego_, UNTAINT(p->I), UNTAINT(p->O));
}

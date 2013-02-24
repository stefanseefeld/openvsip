/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "dft.h"

/* use the apply() operation for DFT problems */
void X(dft_solve)(const plan *ego_, const problem *p_)
{
     const plan_dft *ego = (const plan_dft *) ego_;
     const problem_dft *p = (const problem_dft *) p_;
     ego->apply(ego_, 
		UNTAINT(p->ri), UNTAINT(p->ii), 
		UNTAINT(p->ro), UNTAINT(p->io));
}

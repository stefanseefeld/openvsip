/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "dft.h"

/* guru interface: requires care in alignment etcetera. */
void X(execute_dft)(const X(plan) p, C *in, C *out)
WITH_ALIGNED_STACK({
     plan_dft *pln = (plan_dft *) p->pln;
     if (p->sign == FFT_SIGN)
	  pln->apply((plan *) pln, in[0], in[0]+1, out[0], out[0]+1);
     else
	  pln->apply((plan *) pln, in[0]+1, in[0], out[0]+1, out[0]);
})

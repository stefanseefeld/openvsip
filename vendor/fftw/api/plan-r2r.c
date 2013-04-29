/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

X(plan) X(plan_r2r)(int rank, const int *n, R *in, R *out,
		    const X(r2r_kind) * kind, unsigned flags)
{
     return X(plan_many_r2r)(rank, n, 1, in, 0, 1, 1, out, 0, 1, 1, kind,
			     flags);
}

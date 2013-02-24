/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

X(plan) X(plan_r2r_2d)(int nx, int ny, R *in, R *out,
		       X(r2r_kind) kindx, X(r2r_kind) kindy, unsigned flags)
{
     int n[2];
     X(r2r_kind) kind[2];
     n[0] = nx;
     n[1] = ny;
     kind[0] = kindx;
     kind[1] = kindy;
     return X(plan_r2r)(2, n, in, out, kind, flags);
}

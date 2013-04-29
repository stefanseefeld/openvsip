/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

X(plan) X(plan_r2r_3d)(int nx, int ny, int nz,
		       R *in, R *out, X(r2r_kind) kindx,
		       X(r2r_kind) kindy, X(r2r_kind) kindz, unsigned flags)
{
     int n[3];
     X(r2r_kind) kind[3];
     n[0] = nx;
     n[1] = ny;
     n[2] = nz;
     kind[0] = kindx;
     kind[1] = kindy;
     kind[2] = kindz;
     return X(plan_r2r)(3, n, in, out, kind, flags);
}

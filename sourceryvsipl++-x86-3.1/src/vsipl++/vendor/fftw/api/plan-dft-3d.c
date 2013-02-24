/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "dft.h"

X(plan) X(plan_dft_3d)(int nx, int ny, int nz,
		       C *in, C *out, int sign, unsigned flags)
{
     int n[3];
     n[0] = nx;
     n[1] = ny;
     n[2] = nz;
     return X(plan_dft)(3, n, in, out, sign, flags);
}

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

X(plan) X(plan_dft_r2c_3d)(int nx, int ny, int nz,
			   R *in, C *out, unsigned flags)
{
     int n[3];
     n[0] = nx;
     n[1] = ny;
     n[2] = nz;
     return X(plan_dft_r2c)(3, n, in, out, flags);
}

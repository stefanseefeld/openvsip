/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

X(plan) X(plan_dft_r2c_2d)(int nx, int ny, R *in, C *out, unsigned flags)
{
     int n[2];
     n[0] = nx;
     n[1] = ny;
     return X(plan_dft_r2c)(2, n, in, out, flags);
}

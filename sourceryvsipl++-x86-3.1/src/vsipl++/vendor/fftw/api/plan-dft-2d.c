/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "dft.h"

X(plan) X(plan_dft_2d)(int nx, int ny, C *in, C *out, int sign, unsigned flags)
{
     int n[2];
     n[0] = nx;
     n[1] = ny;
     return X(plan_dft)(2, n, in, out, sign, flags);
}

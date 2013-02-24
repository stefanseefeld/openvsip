/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

X(plan) X(plan_dft_r2c_1d)(int n, R *in, C *out, unsigned flags)
{
     return X(plan_dft_r2c)(1, &n, in, out, flags);
}

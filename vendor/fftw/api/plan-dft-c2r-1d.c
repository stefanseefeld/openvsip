/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

X(plan) X(plan_dft_c2r_1d)(int n, C *in, R *out, unsigned flags)
{
     return X(plan_dft_c2r)(1, &n, in, out, flags);
}

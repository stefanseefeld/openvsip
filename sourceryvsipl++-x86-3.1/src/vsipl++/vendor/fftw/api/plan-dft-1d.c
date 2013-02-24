/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "dft.h"

X(plan) X(plan_dft_1d)(int n, C *in, C *out, int sign, unsigned flags)
{
     return X(plan_dft)(1, &n, in, out, sign, flags);
}

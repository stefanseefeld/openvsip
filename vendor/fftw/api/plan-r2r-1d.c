/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

X(plan) X(plan_r2r_1d)(int n, R *in, R *out, X(r2r_kind) kind, unsigned flags)
{
     return X(plan_r2r)(1, &n, in, out, &kind, flags);
}

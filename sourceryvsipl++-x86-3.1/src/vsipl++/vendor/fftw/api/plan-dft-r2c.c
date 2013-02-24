/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

X(plan) X(plan_dft_r2c)(int rank, const int *n, R *in, C *out, unsigned flags)
{
     return X(plan_many_dft_r2c)(rank, n, 1,
				 in, 0, 1, 1, 
				 out, 0, 1, 1, 
				 flags);
}

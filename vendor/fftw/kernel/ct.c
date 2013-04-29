/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

/* common routines for Cooley-Tukey algorithms */

#include "ifftw.h"

#define POW2P(n) (((n) > 0) && (((n) & ((n) - 1)) == 0))

/* TRUE if radix-r is ugly for size n */
int X(ct_uglyp)(INT min_n, INT n, INT r)
{
     return (n <= min_n) || (POW2P(n) && (n / r) <= 4);
}

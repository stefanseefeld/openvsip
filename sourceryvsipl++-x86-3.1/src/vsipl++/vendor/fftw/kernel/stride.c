/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "ifftw.h"

const INT X(an_INT_guaranteed_to_be_zero) = 0;

#ifdef PRECOMPUTE_ARRAY_INDICES
stride X(mkstride)(INT n, INT s)
{
     int i;
     INT *p = (INT *) MALLOC(n * sizeof(INT), STRIDES);

     for (i = 0; i < n; ++i)
          p[i] = s * i;

     return p;
}

void X(stride_destroy)(stride p)
{
     X(ifree0)(p);
}

#endif

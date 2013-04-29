/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"

int X(tensor_kosherp)(const tensor *x)
{
     int i;

     if (x->rnk < 0) return 0;

     if (FINITE_RNK(x->rnk)) {
	  for (i = 0; i < x->rnk; ++i)
	       if (x->dims[i].n < 0)
		    return 0;
     }
     return 1;
}

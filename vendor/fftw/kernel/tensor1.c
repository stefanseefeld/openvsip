/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"

tensor *X(mktensor_0d)(void)
{
     return X(mktensor(0));
}

tensor *X(mktensor_1d)(INT n, INT is, INT os)
{
     tensor *x = X(mktensor)(1);
     x->dims[0].n = n;
     x->dims[0].is = is;
     x->dims[0].os = os;
     return x;
}

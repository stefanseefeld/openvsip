/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"

INT X(imax)(INT a, INT b)
{
     return (a > b) ? a : b;
}

INT X(imin)(INT a, INT b)
{
     return (a < b) ? a : b;
}

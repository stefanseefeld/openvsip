/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"


void *X(malloc)(size_t n)
{
     return X(kernel_malloc)(n);
}

void X(free)(void *p)
{
     X(kernel_free)(p);
}

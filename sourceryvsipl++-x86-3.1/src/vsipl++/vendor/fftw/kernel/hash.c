/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "ifftw.h"

unsigned X(hash)(const char *s)
{
     unsigned h = 0xDEADBEEFu;
     do {
	  h = h * 17 + (int)*s;
     } while (*s++);
     return h;
}


/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"

void X(tensor_destroy2)(tensor *a, tensor *b)
{
     X(tensor_destroy)(a);
     X(tensor_destroy)(b);
}

void X(tensor_destroy4)(tensor *a, tensor *b, tensor *c, tensor *d)
{
     X(tensor_destroy2)(a, b);
     X(tensor_destroy2)(c, d);
}

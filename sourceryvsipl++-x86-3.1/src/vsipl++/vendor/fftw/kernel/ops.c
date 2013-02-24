/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"

void X(ops_zero)(opcnt *dst)
{
     dst->add = dst->mul = dst->fma = dst->other = 0;
}

void X(ops_cpy)(const opcnt *src, opcnt *dst)
{
     *dst = *src;
}

void X(ops_other)(INT o, opcnt *dst)
{
     X(ops_zero)(dst);
     dst->other = o;
}

void X(ops_madd)(INT m, const opcnt *a, const opcnt *b, opcnt *dst)
{
     dst->add = m * a->add + b->add;
     dst->mul = m * a->mul + b->mul;
     dst->fma = m * a->fma + b->fma;
     dst->other = m * a->other + b->other;
}

void X(ops_add)(const opcnt *a, const opcnt *b, opcnt *dst)
{
     X(ops_madd)(1, a, b, dst);
}

void X(ops_add2)(const opcnt *a, opcnt *dst)
{
     X(ops_add)(a, dst, dst);
}

void X(ops_madd2)(INT m, const opcnt *a, opcnt *dst)
{
     X(ops_madd)(m, a, dst, dst);
}


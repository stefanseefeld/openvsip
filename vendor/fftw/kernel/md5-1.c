/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "ifftw.h"


void X(md5putb)(md5 *p, const void *d_, size_t len)
{
     size_t i;
     const unsigned char *d = (const unsigned char *)d_;
     for (i = 0; i < len; ++i)
	  X(md5putc)(p, d[i]);
}

void X(md5puts)(md5 *p, const char *s)
{
     /* also hash final '\0' */
     do {
	  X(md5putc)(p, *s);
     } while(*s++);
}

void X(md5int)(md5 *p, int i)
{
     X(md5putb)(p, &i, sizeof(i));
}

void X(md5INT)(md5 *p, INT i)
{
     X(md5putb)(p, &i, sizeof(i));
}

void X(md5unsigned)(md5 *p, unsigned i)
{
     X(md5putb)(p, &i, sizeof(i));
}


/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include <stdio.h>

#define BUFSZ 256

typedef struct {
     printer super;
     FILE *f;
     char buf[BUFSZ];
     char *bufw;
} P;

static void myflush(P *p)
{
     fwrite(p->buf, 1, p->bufw - p->buf, p->f);
     p->bufw = p->buf;
}

static void myputchr(printer *p_, char c)
{
     P *p = (P *) p_;
     if (p->bufw >= p->buf + BUFSZ)
	  myflush(p);
     *p->bufw++ = c;
}

static void mycleanup(printer *p_)
{
     P *p = (P *) p_;
     myflush(p);
}

printer *X(mkprinter_file)(FILE *f)
{
     P *p = (P *) X(mkprinter)(sizeof(P), myputchr, mycleanup);
     p->f = f;
     p->bufw = p->buf;
     return &p->super;
}

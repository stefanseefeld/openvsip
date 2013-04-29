/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "ifftw.h"

#ifdef FFTW_DEBUG
#include <stdio.h>

typedef struct {
     printer super;
     FILE *f;
} P_file;

static void putchr_file(printer *p_, char c)
{
     P_file *p = (P_file *) p_;
     fputc(c, p->f);
}

static printer *mkprinter_file(FILE *f)
{
     P_file *p = (P_file *) X(mkprinter)(sizeof(P_file), putchr_file, 0);
     p->f = f;
     return &p->super;
}

void X(debug)(const char *format, ...)
{
     va_list ap;
     printer *p = mkprinter_file(stderr);
     va_start(ap, format);
     p->vprint(p, format, ap);
     va_end(ap);
     X(printer_destroy)(p);
}
#endif

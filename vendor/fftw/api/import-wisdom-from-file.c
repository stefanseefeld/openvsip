/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include <stdio.h>

/* getc()/putc() are *unbelievably* slow on linux.  Looks like glibc
   is grabbing a lock for each call to getc()/putc(), or something
   like that.  You pay the price for these idiotic posix threads
   whether you use them or not.

   So, we do our own buffering.  This completely defeats the purpose
   of having stdio in the first place, of course.
*/
  
#define BUFSZ 256

typedef struct {
     scanner super;
     FILE *f;
     char buf[BUFSZ];
     char *bufr, *bufw;
} S;

static int getchr_file(scanner * sc_)
{
     S *sc = (S *) sc_;

     if (sc->bufr >= sc->bufw) {
	  sc->bufr = sc->buf;
	  sc->bufw = sc->buf + fread(sc->buf, 1, BUFSZ, sc->f);
	  if (sc->bufr >= sc->bufw)
	       return EOF;
     }

     return *(sc->bufr++);
}

static scanner *mkscanner_file(FILE *f)
{
     S *sc = (S *) X(mkscanner)(sizeof(S), getchr_file);
     sc->f = f;
     sc->bufr = sc->bufw = sc->buf;
     return &sc->super;
}

int X(import_wisdom_from_file)(FILE *input_file)
{
     scanner *s = mkscanner_file(input_file);
     planner *plnr = X(the_planner)();
     int ret = plnr->adt->imprt(plnr, s);
     X(scanner_destroy)(s);
     return ret;
}

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

typedef struct {
     printer super;
     void (*write_char)(char c, void *);
     void *data;
} P;

static void putchr_generic(printer * p_, char c)
{
     P *p = (P *) p_;
     (p->write_char)(c, p->data);
}

void X(export_wisdom)(void (*write_char)(char c, void *), void *data)
{
     P *p = (P *) X(mkprinter)(sizeof(P), putchr_generic, 0);
     planner *plnr = X(the_planner)();

     p->write_char = write_char;
     p->data = data;
     plnr->adt->exprt(plnr, (printer *) p);
     X(printer_destroy)((printer *) p);
}

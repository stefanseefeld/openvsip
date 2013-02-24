/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

typedef struct {
     scanner super;
     int (*read_char)(void *);
     void *data;
} S;

static int getchr_generic(scanner * s_)
{
     S *s = (S *) s_;
     return (s->read_char)(s->data);
}

int X(import_wisdom)(int (*read_char)(void *), void *data)
{
     S *s = (S *) X(mkscanner)(sizeof(S), getchr_generic);
     planner *plnr = X(the_planner)();
     int ret;

     s->read_char = read_char;
     s->data = data;
     ret = plnr->adt->imprt(plnr, (scanner *) s);
     X(scanner_destroy)((scanner *) s);
     return ret;
}

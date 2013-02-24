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
     const char *s;
} S_str;

static int getchr_str(scanner * sc_)
{
     S_str *sc = (S_str *) sc_;
     if (!*sc->s)
          return EOF;
     return *sc->s++;
}

static scanner *mkscanner_str(const char *s)
{
     S_str *sc = (S_str *) X(mkscanner)(sizeof(S_str), getchr_str);
     sc->s = s;
     return &sc->super;
}

int X(import_wisdom_from_string)(const char *input_string)
{
     scanner *s = mkscanner_str(input_string);
     planner *plnr = X(the_planner)();
     int ret = plnr->adt->imprt(plnr, s);
     X(scanner_destroy)(s);
     return ret;
}

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
     int *cnt;
} P_cnt;

static void putchr_cnt(printer * p_, char c)
{
     P_cnt *p = (P_cnt *) p_;
     UNUSED(c);
     ++*p->cnt;
}

static printer *mkprinter_cnt(int *cnt)
{
     P_cnt *p = (P_cnt *) X(mkprinter)(sizeof(P_cnt), putchr_cnt, 0);
     p->cnt = cnt;
     *cnt = 0;
     return &p->super;
}

typedef struct {
     printer super;
     char *s;
} P_str;

static void putchr_str(printer * p_, char c)
{
     P_str *p = (P_str *) p_;
     *p->s++ = c;
     *p->s = 0;
}

static printer *mkprinter_str(char *s)
{
     P_str *p = (P_str *) X(mkprinter)(sizeof(P_str), putchr_str, 0);
     p->s = s;
     *s = 0;
     return &p->super;
}

char *X(export_wisdom_to_string)(void)
{
     printer *p;
     planner *plnr = X(the_planner)();
     int cnt;
     char *s;

     p = mkprinter_cnt(&cnt);
     plnr->adt->exprt(plnr, p);
     X(printer_destroy)(p);

     s = (char *) NATIVE_MALLOC(sizeof(char) * (cnt + 1), OTHER);
     if (s) {
          p = mkprinter_str(s);
          plnr->adt->exprt(plnr, p);
          X(printer_destroy)(p);
     }

     return s;
}

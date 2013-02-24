/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

void X(fprint_plan)(const X(plan) p, FILE *output_file)
{
     printer *pr = X(mkprinter_file)(output_file);
     plan *pln = p->pln;
     pln->adt->print(pln, pr);
     X(printer_destroy)(pr);
}

void X(print_plan)(const X(plan) p)
{
     X(fprint_plan)(p, stdout);
}

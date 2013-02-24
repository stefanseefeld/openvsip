/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

void X(export_wisdom_to_file)(FILE *output_file)
{
     printer *p = X(mkprinter_file)(output_file);
     planner *plnr = X(the_planner)();
     plnr->adt->exprt(plnr, p);
     X(printer_destroy)(p);
}

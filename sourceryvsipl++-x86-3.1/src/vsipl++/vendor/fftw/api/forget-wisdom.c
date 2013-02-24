/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

void X(forget_wisdom)(void)
{
     planner *plnr = X(the_planner)();
     plnr->adt->forget(plnr, FORGET_EVERYTHING);
}

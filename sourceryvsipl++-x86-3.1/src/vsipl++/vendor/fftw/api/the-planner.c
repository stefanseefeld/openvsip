/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

static planner *plnr = 0;

/* create the planner for the rest of the API */
planner *X(the_planner)(void)
{
     if (!plnr) {
          plnr = X(mkplanner)();
          X(configure_planner)(plnr);
     }

     return plnr;
}

void X(cleanup)(void)
{
     if (plnr) {
          X(planner_destroy)(plnr);
          plnr = 0;
     }
}

void X(set_timelimit)(double tlim) 
{
     /* PLNR is not necessarily initialized when this function is
	called, so use X(the_planner)() */
     X(the_planner)()->timelimit = tlim; 
}

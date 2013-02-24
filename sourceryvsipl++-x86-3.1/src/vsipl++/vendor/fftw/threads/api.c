/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"
#include "threads.h"

static int threads_inited = 0;

static void threads_register_hooks(void)
{
     X(mksolver_ct_hook) = X(mksolver_ct_threads);
     X(mksolver_hc2hc_hook) = X(mksolver_hc2hc_threads);
}

static void threads_unregister_hooks(void)
{
     X(mksolver_ct_hook) = 0;
     X(mksolver_hc2hc_hook) = 0;
}

/* should be called before all other FFTW functions! */
int X(init_threads)(void)
{
     if (!threads_inited) {
	  planner *plnr;

          if (X(ithreads_init)())
               return 0;

	  threads_register_hooks();

	  /* this should be the first time the_planner is called,
	     and hence the time it is configured */
	  plnr = X(the_planner)();
	  X(threads_conf_standard)(plnr);
	       
          threads_inited = 1;
     }
     return 1;
}


void X(cleanup_threads)(void)
{
     X(cleanup)();
     if (threads_inited) {
	  X(threads_cleanup)();
	  threads_unregister_hooks();
	  threads_inited = 0;
     }
}

void X(plan_with_nthreads)(int nthreads)
{
     planner *plnr;

     if (!threads_inited) {
	  X(cleanup)();
	  X(init_threads)();
     }
     A(threads_inited);
     plnr = X(the_planner)();
     plnr->nthr = X(imax)(1, nthreads);
}

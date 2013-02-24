/*
 * Copyright (c) 2001 Matteo Frigo
 * Copyright (c) 2001 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "bench.h"

int no_speed_allocation = 0; /* 1 to not allocate array data in speed() */

void speed(const char *param, int setup_only)
{
     double *t;
     int iter = 0, k;
     bench_problem *p;
     double tmin, y;

     t = (double *) bench_malloc(time_repeat * sizeof(double));

     for (k = 0; k < time_repeat; ++k) 
	  t[k] = 0;

     p = problem_parse(param);
     BENCH_ASSERT(can_do(p));
     if (!no_speed_allocation) {
	  problem_alloc(p);
	  problem_zero(p);
     }

     timer_start(LIBBENCH_TIMER);
     setup(p);
     p->setup_time = bench_cost_postprocess(timer_stop(LIBBENCH_TIMER));

     /* reset the input to zero again, because the planner in paranoid
	mode sets it to random values, thus making the benchmark
	diverge. */
     if (!no_speed_allocation) 
	  problem_zero(p);
     
     if (setup_only)
	  goto done;

 start_over:
     for (iter = 1; iter < (1<<30); iter *= 2) {
	  tmin = 1.0e20;
	  for (k = 0; k < time_repeat; ++k) {
	       timer_start(LIBBENCH_TIMER);
	       doit(iter, p);
	       y = bench_cost_postprocess(timer_stop(LIBBENCH_TIMER));
	       if (y < 0) /* yes, it happens */
		    goto start_over;
	       t[k] = y;
	       if (y < tmin)
		    tmin = y;
	  }
	  
	  if (tmin >= time_min)
	       goto done;
     }

     goto start_over; /* this also happens */

 done:
     done(p);

     if (iter) 
	  for (k = 0; k < time_repeat; ++k) 
	       t[k] /= iter;
     else
	  for (k = 0; k < time_repeat; ++k) 
	       t[k] = 0;

     report(p, t, time_repeat);

     if (!no_speed_allocation)
	  problem_destroy(p);
     bench_free(t);
     return;
}

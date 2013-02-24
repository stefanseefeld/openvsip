/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

/* openmp.c: thread spawning via OpenMP  */

#include "threads.h"

#ifdef HAVE_OPENMP

#if !defined(_OPENMP)
#error OpenMP enabled but not using an OpenMP compiler
#endif

int X(ithreads_init)(void)
{
     return 0; /* no error */
}

/* Distribute a loop from 0 to loopmax-1 over nthreads threads.
   proc(d) is called to execute a block of iterations from d->min
   to d->max-1.  d->thr_num indicate the number of the thread
   that is executing proc (from 0 to nthreads-1), and d->data is
   the same as the data parameter passed to X(spawn_loop).

   This function returns only after all the threads have completed. */
void X(spawn_loop)(int loopmax, int nthr, spawn_function proc, void *data)
{
     int block_size;
     spawn_data d;
     int i;

     A(loopmax >= 0);
     A(nthr > 0);
     A(proc);

     if (!loopmax) return;

     /* Choose the block size and number of threads in order to (1)
        minimize the critical path and (2) use the fewest threads that
        achieve the same critical path (to minimize overhead).
        e.g. if loopmax is 5 and nthr is 4, we should use only 3
        threads with block sizes of 2, 2, and 1. */
     block_size = (loopmax + nthr - 1) / nthr;
     nthr = (loopmax + block_size - 1) / block_size;

     THREAD_ON; /* prevent debugging mode from failing under threads */
#pragma omp parallel for private(d)
     for (i = 0; i < nthr; ++i) {
	  d.max = (d.min = i * block_size) + block_size;
	  if (d.max > loopmax)
	       d.max = loopmax;
	  d.thr_num = i;
	  d.data = data;
	  proc(&d);
     }
     THREAD_OFF; /* prevent debugging mode from failing under threads */
}

void X(threads_cleanup)(void)
{
}

#endif /* HAVE_OPENMP */

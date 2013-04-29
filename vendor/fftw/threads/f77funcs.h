/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

/* Functions in the FFTW Fortran API, mangled according to the
   F77(...) macro.  This file is designed to be #included by
   f77api.c, possibly multiple times in order to support multiple
   compiler manglings (via redefinition of F77). */

FFTW_VOIDFUNC F77(plan_with_nthreads, PLAN_WITH_NTHREADS)(int *nthreads)
{
     X(plan_with_nthreads)(*nthreads);
}

FFTW_VOIDFUNC F77(init_threads, INIT_THREADS)(int *okay)
{
     *okay = X(init_threads)();
}

FFTW_VOIDFUNC F77(cleanup_threads, CLEANUP_THREADS)(void)
{
     X(cleanup_threads)();
}

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "threads.h"

static const solvtab s =
{
#if defined(HAVE_THREADS) || defined(HAVE_OPENMP)

     SOLVTAB(X(dft_thr_vrank_geq1_register)),
     SOLVTAB(X(rdft_thr_vrank_geq1_register)),
     SOLVTAB(X(rdft2_thr_vrank_geq1_register)),

#endif 

     SOLVTAB_END
};

void X(threads_conf_standard)(planner *p)
{
     X(solvtab_exec)(s, p);
}

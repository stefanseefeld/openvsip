/*
 * Copyright (c) 2001 Matteo Frigo
 * Copyright (c) 2001 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "bench.h"

/* set I/O arrays to zero.  Default routine */
void problem_zero(bench_problem *p)
{
     bench_complex czero = {0, 0};
     if (p->kind == PROBLEM_COMPLEX) {
	  caset((bench_complex *) p->inphys, p->iphyssz, czero);
	  caset((bench_complex *) p->outphys, p->ophyssz, czero);
     } else if (p->kind == PROBLEM_R2R) {
	  aset((bench_real *) p->inphys, p->iphyssz, 0.0);
	  aset((bench_real *) p->outphys, p->ophyssz, 0.0);
     } else if (p->kind == PROBLEM_REAL && p->sign < 0) {
	  aset((bench_real *) p->inphys, p->iphyssz, 0.0);
	  caset((bench_complex *) p->outphys, p->ophyssz, czero);
     } else if (p->kind == PROBLEM_REAL && p->sign > 0) {
	  caset((bench_complex *) p->inphys, p->iphyssz, czero);
	  aset((bench_real *) p->outphys, p->ophyssz, 0.0);
     } else {
	  BENCH_ASSERT(0); /* TODO */
     }
}

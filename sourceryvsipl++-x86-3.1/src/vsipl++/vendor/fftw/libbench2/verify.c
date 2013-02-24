/*
 * Copyright (c) 2000 Matteo Frigo
 * Copyright (c) 2000 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include <stdio.h>
#include <stdlib.h>

#include "verify.h"

void verify_problem(bench_problem *p, int rounds, double tol)
{
     errors e;
     const char *pstring = p->pstring ? p->pstring : "<unknown problem>";

     switch (p->kind) {
	 case PROBLEM_COMPLEX: verify_dft(p, rounds, tol, &e); break;
	 case PROBLEM_REAL: verify_rdft2(p, rounds, tol, &e); break;
	 case PROBLEM_R2R: verify_r2r(p, rounds, tol, &e); break;
     }

     if (verbose)
	  ovtpvt("%s %g %g %g\n", pstring, e.l, e.i, e.s);
}

void verify(const char *param, int rounds, double tol)
{
     bench_problem *p;

     p = problem_parse(param);
     if (!can_do(p)) {
	  ovtpvt_err("No can_do for %s\n", p->pstring);
	  BENCH_ASSERT(0);
     }
     problem_alloc(p);
     problem_zero(p);
     setup(p);

     verify_problem(p, rounds, tol);

     done(p);
     problem_destroy(p);
}


static void do_accuracy(bench_problem *p, int rounds, int impulse_rounds)
{
     double t[6];

     switch (p->kind) {
	 case PROBLEM_COMPLEX:
	      accuracy_dft(p, rounds, impulse_rounds, t); break;
	 case PROBLEM_REAL:
	      accuracy_rdft2(p, rounds, impulse_rounds, t); break;
	 case PROBLEM_R2R:
	      accuracy_r2r(p, rounds, impulse_rounds, t); break;
     }

     /* t[0] : L1 error
	t[1] : L2 error
	t[2] : Linf error
	t[3..5]: L1, L2, Linf backward error */
     ovtpvt("%6.2e %6.2e %6.2e %6.2e %6.2e %6.2e\n", 
	    t[0], t[1], t[2], t[3], t[4], t[5]);
}

void accuracy(const char *param, int rounds, int impulse_rounds)
{
     bench_problem *p;
     p = problem_parse(param);
     BENCH_ASSERT(can_do(p));
     problem_alloc(p);
     problem_zero(p);
     setup(p);
     do_accuracy(p, rounds, impulse_rounds);
     done(p);
     problem_destroy(p);
}

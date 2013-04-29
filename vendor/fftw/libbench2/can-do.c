/*
 * Copyright (c) 2001 Matteo Frigo
 * Copyright (c) 2001 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "bench.h"
#include <stdio.h>

void report_can_do(const char *param)
{
     bench_problem *p;
     p = problem_parse(param);
     ovtpvt("#%c\n", can_do(p) ? 't' : 'f');
     problem_destroy(p);
}

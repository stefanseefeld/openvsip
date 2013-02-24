/*
 * Copyright (c) 2001 Matteo Frigo
 * Copyright (c) 2001 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "bench.h"
#include <stdio.h>
#include <string.h>

void report_info(const char *param)
{
     struct bench_doc *p;

     for (p = bench_doc; p->key; ++p) {
	  if (!strcmp(param, p->key)) {
	       if (!p->val)
		    p->val = p->f();

	       ovtpvt("%s\n", p->val);
	  }
     }
}

void report_info_all(void)
{
     struct bench_doc *p;

     /*
      * TODO: escape quotes?  The format is not unambigously
      * parseable if the info string contains double quotes.
      */
     for (p = bench_doc; p->key; ++p) {
	  if (!p->val)
	       p->val = p->f();
	  ovtpvt("(%s \"%s\")\n", p->key, p->val);
     }
     ovtpvt("(benchmark-precision \"%s\")\n", 
	    SINGLE_PRECISION ? "single" : 
	    (LDOUBLE_PRECISION ? "long-double" : "double"));
}


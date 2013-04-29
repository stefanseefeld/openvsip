/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "ifftw.h"
#include <stdio.h>
#include <stdlib.h>

void X(assertion_failed)(const char *s, int line, const char *file)
{
     fflush(stdout);
     fprintf(stderr, "fftw: %s:%d: assertion failed: %s\n", file, line, s);
#ifdef HAVE_ABORT
     abort();
#else
     exit(EXIT_FAILURE);
#endif
}

/*
 * Copyright (c) 2001 Matteo Frigo
 * Copyright (c) 2001 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "bench.h"

/* On some systems, we are required to define a dummy main-like
   routine (called "MAIN__" or something similar in order to link a C
   main() with the Fortran libraries).  This is detected by autoconf;
   see the autoconf 2.52 or later manual. */
#ifdef F77_DUMMY_MAIN
#  ifdef __cplusplus
     extern "C"
#  endif
     int F77_DUMMY_MAIN() { return 1; }
#endif

/* in a separate file so that the user can override it */
int main(int argc, char *argv[])
{
     return aligned_main(argc, argv);
}

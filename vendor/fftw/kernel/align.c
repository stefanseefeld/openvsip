/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"

#if HAVE_SIMD || HAVE_CELL
#  define ALGN 16
#else
   /* disable the alignment machinery, because it will break,
      e.g., if sizeof(R) == 12 (as in long-double/x86) */
#  define ALGN 0
#endif

/* NONPORTABLE */
int X(alignment_of)(R *p)
{
#if ALGN == 0
     UNUSED(p);
     return 0;
#else
     return (int)(((uintptr_t) p) % ALGN);
#endif
}

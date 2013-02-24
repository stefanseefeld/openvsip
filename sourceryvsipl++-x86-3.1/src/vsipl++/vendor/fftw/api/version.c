/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "api.h"

const char X(cc)[] = FFTW_CC;
const char X(codelet_optim)[] = CODELET_OPTIM;

const char X(version)[] = PACKAGE "-" PACKAGE_VERSION

#if HAVE_FMA
   "-fma"
#endif

#if HAVE_SSE
   "-sse"
#endif

#if HAVE_SSE2
   "-sse2"
#endif

#if HAVE_ALTIVEC
   "-altivec"
#endif

;

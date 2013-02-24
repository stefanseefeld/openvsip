/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "simd.h"

#undef LD
#define LD LDA
#undef ST
#define ST STA

#define VTW VTW3
#define TWVL TWVL3
#define LDW(x) LDA(x, 0, 0) /* load twiddle factor */

/* same as t1f otherwise */
#define GENUS X(dft_t1fsimd_genus)
extern const ct_genus GENUS;


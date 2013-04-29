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

#define VTW VTWS
#define TWVL TWVLS
#define LDW(x) LDA(x, 0, 0) /* load twiddle factor */

#define GENUS X(dft_tssimd_genus)
extern const ct_genus GENUS;


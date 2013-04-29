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

#define VTW VTW1
#define TWVL TWVL1
#define BYTW BYTW1
#define BYTWJ BYTWJ1

#define GENUS X(dft_t1bsimd_genus)
extern const ct_genus GENUS;


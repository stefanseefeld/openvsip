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

#define VTW VTW2
#define TWVL TWVL2
#define BYTW BYTW2
#define BYTWJ BYTWJ2

#define GENUS X(dft_t2bsimd_genus)
extern const ct_genus GENUS;


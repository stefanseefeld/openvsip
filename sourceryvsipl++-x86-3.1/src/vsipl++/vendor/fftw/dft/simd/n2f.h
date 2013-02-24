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

#define GENUS X(dft_n2fsimd_genus)
extern const kdft_genus GENUS;

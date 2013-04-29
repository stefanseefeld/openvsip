/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "simd.h"

#define VTW VTW3
#define TWVL TWVL3
#define LDW(x) LDA(x, 0, 0)

#define GENUS X(rdft_hc2cfv_genus)
extern const hc2c_genus GENUS;

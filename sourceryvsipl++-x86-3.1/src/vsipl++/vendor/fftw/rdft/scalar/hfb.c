/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-rdft.h"
#include "hf.h"

const hc2hc_genus GENUS = { R2HC, 1 };

#undef GENUS
#include "hb.h"

const hc2hc_genus GENUS = { HC2R, 1 };

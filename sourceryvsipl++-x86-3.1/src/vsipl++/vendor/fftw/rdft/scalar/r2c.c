/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-rdft.h"

#include "r2cf.h"
const kr2c_genus GENUS = { R2HC, 1 };
#undef GENUS

#include "r2cfII.h"
const kr2c_genus GENUS = { R2HCII, 1 };
#undef GENUS

#include "r2cb.h"
const kr2c_genus GENUS = { HC2R, 1 };
#undef GENUS

#include "r2cbIII.h"
const kr2c_genus GENUS = { HC2RIII, 1 };
#undef GENUS

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "rdft.h"

void X(kr2r_register)(planner *p, kr2r codelet, const kr2r_desc *desc)
{
     REGISTER_SOLVER(p, X(mksolver_rdft_r2r_direct)(codelet, desc));
}

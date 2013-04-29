/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "rdft.h"

void X(kr2c_register)(planner *p, kr2c codelet, const kr2c_desc *desc)
{
     REGISTER_SOLVER(p, X(mksolver_rdft_r2c_direct)(codelet, desc));
     REGISTER_SOLVER(p, X(mksolver_rdft_r2c_directbuf)(codelet, desc));
     REGISTER_SOLVER(p, X(mksolver_rdft2_direct)(codelet, desc));
}

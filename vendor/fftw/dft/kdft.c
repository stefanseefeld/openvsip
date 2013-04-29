/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "dft.h"

void X(kdft_register)(planner *p, kdft codelet, const kdft_desc *desc)
{
     REGISTER_SOLVER(p, X(mksolver_dft_direct)(codelet, desc));
     REGISTER_SOLVER(p, X(mksolver_dft_directbuf)(codelet, desc));
}

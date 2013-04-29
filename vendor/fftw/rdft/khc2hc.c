/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "hc2hc.h"

void X(khc2hc_register)(planner *p, khc2hc codelet, const hc2hc_desc *desc)
{
     X(regsolver_hc2hc_direct)(p, codelet, desc);
}

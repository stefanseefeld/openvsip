/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ct-hc2c.h"

void X(khc2c_register)(planner *p, khc2c codelet, const hc2c_desc *desc,
		       hc2c_kind hc2ckind)
{
     X(regsolver_hc2c_direct)(p, codelet, desc, hc2ckind);
}

/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ct.h"

void X(kdft_difsq_register)(planner *p, kdftwsq k, const ct_desc *desc)
{
     X(regsolver_ct_directwsq)(p, k, desc, DECDIF);
}

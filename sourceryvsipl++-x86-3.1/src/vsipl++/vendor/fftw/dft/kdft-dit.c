/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ct.h"

void X(kdft_dit_register)(planner *p, kdftw codelet, const ct_desc *desc)
{
     X(regsolver_ct_directw)(p, codelet, desc, DECDIT);
}

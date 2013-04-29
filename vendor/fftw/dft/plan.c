/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "dft.h"

plan *X(mkplan_dft)(size_t size, const plan_adt *adt, dftapply apply)
{
     plan_dft *ego;

     ego = (plan_dft *) X(mkplan)(size, adt);
     ego->apply = apply;

     return &(ego->super);
}

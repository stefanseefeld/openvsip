/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "rdft.h"

plan *X(mkplan_rdft)(size_t size, const plan_adt *adt, rdftapply apply)
{
     plan_rdft *ego;

     ego = (plan_rdft *) X(mkplan)(size, adt);
     ego->apply = apply;

     return &(ego->super);
}

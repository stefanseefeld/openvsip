/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"

void X(solvtab_exec)(const solvtab tbl, planner *p)
{
     for (; tbl->reg_nam; ++tbl) {
	  p->cur_reg_nam = tbl->reg_nam;
	  p->cur_reg_id = 0;
	  tbl->reg(p);
     }
     p->cur_reg_nam = 0;
}


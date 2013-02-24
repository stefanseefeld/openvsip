/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "reodft.h"

static const solvtab s =
{
#if 0 /* 1 to enable "standard" algorithms with substandard accuracy;
         you must also add them to Makefile.am to compile these files*/
     SOLVTAB(X(redft00e_r2hc_register)),
     SOLVTAB(X(rodft00e_r2hc_register)),
     SOLVTAB(X(reodft11e_r2hc_register)),
#endif
     SOLVTAB(X(redft00e_r2hc_pad_register)),
     SOLVTAB(X(rodft00e_r2hc_pad_register)),
     SOLVTAB(X(reodft00e_splitradix_register)),
     SOLVTAB(X(reodft010e_r2hc_register)),
     SOLVTAB(X(reodft11e_radix2_r2hc_register)),
     SOLVTAB(X(reodft11e_r2hc_odd_register)),

     SOLVTAB_END
};

void X(reodft_conf_standard)(planner *p)
{
     X(solvtab_exec)(s, p);
}

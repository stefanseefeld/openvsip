/*
 * Copyright (c) 2007 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "dft.h"

#if HAVE_CELL

#include "fftw-cell.h"

static const solvtab s =
{
     SOLVTAB(X(dft_direct_cell_register)),
     SOLVTAB(X(ct_cell_direct_register)),
     SOLVTAB_END
};

void X(dft_conf_cell)(planner *p)
{
     X(solvtab_exec)(s, p);
}

#endif /* HAVE_CELL */

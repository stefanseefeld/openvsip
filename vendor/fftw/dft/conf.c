/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "dft.h"

static const solvtab s =
{
     SOLVTAB(X(dft_indirect_register)),
     SOLVTAB(X(dft_indirect_transpose_register)),
     SOLVTAB(X(dft_rank_geq2_register)),
     SOLVTAB(X(dft_vrank_geq1_register)),
     SOLVTAB(X(dft_buffered_register)),
     SOLVTAB(X(dft_generic_register)),
     SOLVTAB(X(dft_rader_register)),
     SOLVTAB(X(dft_bluestein_register)),
     SOLVTAB(X(dft_nop_register)),
     SOLVTAB(X(ct_generic_register)),
     SOLVTAB(X(ct_genericbuf_register)),
     SOLVTAB_END
};

void X(dft_conf_standard)(planner *p)
{
     X(solvtab_exec)(s, p);
     X(solvtab_exec)(X(solvtab_dft_standard), p);
#if HAVE_SIMD
     X(solvtab_exec)(X(solvtab_dft_simd), p);
#endif
}

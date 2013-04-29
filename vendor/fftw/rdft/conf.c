/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "rdft.h"

static const solvtab s =
{
     SOLVTAB(X(rdft_indirect_register)),
     SOLVTAB(X(rdft_rank0_register)),
     SOLVTAB(X(rdft_vrank3_transpose_register)),
     SOLVTAB(X(rdft_vrank_geq1_register)),

     SOLVTAB(X(rdft_nop_register)),
     SOLVTAB(X(rdft_buffered_register)),
     SOLVTAB(X(rdft_generic_register)),
     SOLVTAB(X(rdft_rank_geq2_register)),

     SOLVTAB(X(dft_r2hc_register)),

     SOLVTAB(X(rdft_dht_register)),
     SOLVTAB(X(dht_r2hc_register)),
     SOLVTAB(X(dht_rader_register)),

     SOLVTAB(X(rdft2_vrank_geq1_register)),
     SOLVTAB(X(rdft2_nop_register)),
     SOLVTAB(X(rdft2_rank0_register)),
     SOLVTAB(X(rdft2_buffered_register)),
     SOLVTAB(X(rdft2_rank_geq2_register)),
     SOLVTAB(X(rdft2_rdft_register)),

     SOLVTAB(X(hc2hc_generic_register)),

     SOLVTAB_END
};

void X(rdft_conf_standard)(planner *p)
{
     X(solvtab_exec)(s, p);
     X(solvtab_exec)(X(solvtab_rdft_r2cf), p);
     X(solvtab_exec)(X(solvtab_rdft_r2cb), p);
     X(solvtab_exec)(X(solvtab_rdft_r2r), p);

#if HAVE_SIMD
     X(solvtab_exec)(X(solvtab_rdft_simd), p);
#endif
}

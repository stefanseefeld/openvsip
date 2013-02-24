/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "rdft.h"

typedef void (*hc2hcapply) (const plan *ego, R *IO);
typedef struct hc2hc_solver_s hc2hc_solver;
typedef plan *(*hc2hc_mkinferior)(const hc2hc_solver *ego,
			    rdft_kind kind, INT r, INT m, INT s, 
			    INT vl, INT vs, INT mstart, INT mcount,
			    R *IO, planner *plnr);

typedef struct {
     plan super;
     hc2hcapply apply;
} plan_hc2hc;

extern plan *X(mkplan_hc2hc)(size_t size, const plan_adt *adt, 
			     hc2hcapply apply);

#define MKPLAN_HC2HC(type, adt, apply) \
  (type *)X(mkplan_hc2hc)(sizeof(type), adt, apply)

struct hc2hc_solver_s {
     solver super;
     INT r;

     hc2hc_mkinferior mkcldw;
};

hc2hc_solver *X(mksolver_hc2hc)(size_t size, INT r, hc2hc_mkinferior mkcldw);
extern hc2hc_solver *(*X(mksolver_hc2hc_hook))(size_t, INT, hc2hc_mkinferior);

void X(regsolver_hc2hc_direct)(planner *plnr, khc2hc codelet, 
			       const hc2hc_desc *desc);

int X(hc2hc_applicable)(const hc2hc_solver *, const problem *, planner *);

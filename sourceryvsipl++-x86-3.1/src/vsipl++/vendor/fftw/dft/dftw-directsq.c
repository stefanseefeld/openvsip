/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ct.h"

typedef struct {
     ct_solver super;
     const ct_desc *desc;
     kdftwsq k;
} S;

typedef struct {
     plan_dftw super;
     kdftwsq k;
     INT r;
     stride rs, vs;
     INT m, ms, v, mb, me;
     twid *td;
     const S *slv;
} P;


static void apply(const plan *ego_, R *rio, R *iio)
{
     const P *ego = (const P *) ego_;
     INT mb = ego->mb, ms = ego->ms;
     ego->k(rio + mb*ms, iio + mb*ms, ego->td->W, ego->rs, ego->vs,
	    mb, ego->me, ms);
}

static void awake(plan *ego_, enum wakefulness wakefulness)
{
     P *ego = (P *) ego_;

     X(twiddle_awake)(wakefulness, &ego->td, ego->slv->desc->tw,
		      ego->r * ego->m, ego->r, ego->m);
}

static void destroy(plan *ego_)
{
     P *ego = (P *) ego_;
     X(stride_destroy)(ego->rs);
     X(stride_destroy)(ego->vs);
}

static void print(const plan *ego_, printer *p)
{
     const P *ego = (const P *) ego_;
     const S *slv = ego->slv;
     const ct_desc *e = slv->desc;

     p->print(p, "(dftw-directsq-%D/%D%v \"%s\")",
	      ego->r, X(twiddle_length)(ego->r, e->tw), ego->v, e->nam);
}

static int applicable(const S *ego,
		      INT r, INT irs, INT ors,
		      INT m, INT ms,
		      INT v, INT ivs, INT ovs,
		      INT mb, INT me,
		      R *rio, R *iio,
		      const planner *plnr)
{
     const ct_desc *e = ego->desc;
     UNUSED(v);

     return (
	  1
	  && r == e->radix

	  /* transpose r, v */
	  && r == v
	  && irs == ovs
	  && ivs == ors

	  /* check for alignment/vector length restrictions */
	  && e->genus->okp(e, rio, iio, irs, ivs, m, mb, me, ms, plnr)

	  );
}

static plan *mkcldw(const ct_solver *ego_,
		    INT r, INT irs, INT ors,
		    INT m, INT ms,
		    INT v, INT ivs, INT ovs,
		    INT mstart, INT mcount,
		    R *rio, R *iio,
		    planner *plnr)
{
     const S *ego = (const S *) ego_;
     P *pln;
     const ct_desc *e = ego->desc;

     static const plan_adt padt = {
	  0, awake, print, destroy
     };

     A(mstart >= 0 && mstart + mcount <= m);
     if (!applicable(ego,
		     r, irs, ors, m, ms, v, ivs, ovs, mstart, mstart + mcount,
		     rio, iio, plnr))
          return (plan *)0;

     pln = MKPLAN_DFTW(P, &padt, apply);

     pln->k = ego->k;
     pln->rs = X(mkstride)(r, irs);
     pln->vs = X(mkstride)(v, ivs);
     pln->td = 0;
     pln->r = r;
     pln->m = m;
     pln->ms = ms;
     pln->v = v;
     pln->mb = mstart;
     pln->me = mstart + mcount;
     pln->slv = ego;

     X(ops_zero)(&pln->super.super.ops);
     X(ops_madd2)(mcount/e->genus->vl, &e->ops, &pln->super.super.ops);

     return &(pln->super.super);
}

static void regone(planner *plnr, kdftwsq codelet,
		   const ct_desc *desc, int dec)
{
     S *slv = (S *)X(mksolver_ct)(sizeof(S), desc->radix, dec, mkcldw, 0);
     slv->k = codelet;
     slv->desc = desc;
     REGISTER_SOLVER(plnr, &(slv->super.super));
     if (X(mksolver_ct_hook)) {
	  slv = (S *)X(mksolver_ct_hook)(sizeof(S), desc->radix, dec,
					 mkcldw, 0);
	  slv->k = codelet;
	  slv->desc = desc;
	  REGISTER_SOLVER(plnr, &(slv->super.super));
     }
}

void X(regsolver_ct_directwsq)(planner *plnr, kdftwsq codelet,
			       const ct_desc *desc, int dec)
{
     regone(plnr, codelet, desc, dec+TRANSPOSE);
}

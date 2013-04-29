/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "codelet-dft.h"

#if HAVE_SIMD

#include "simd.h"

static int okp_common(const ct_desc *d,
		      const R *rio, const R *iio, 
		      INT rs, INT vs, INT m, INT mb, INT me, INT ms,
		      const planner *plnr)
{
     UNUSED(rio); UNUSED(iio);
     return (RIGHT_CPU()
	     && !NO_SIMDP(plnr)
	     && SIMD_STRIDE_OKA(rs)
	     && SIMD_VSTRIDE_OKA(ms)
             && (m % VL) == 0
             && (mb % VL) == 0
             && (me % VL) == 0
	     && (!d->rs || (d->rs == rs))
	     && (!d->vs || (d->vs == vs))
	     && (!d->ms || (d->ms == ms))
	  );
}

static int okp_commonu(const ct_desc *d,
		      const R *rio, const R *iio, 
		      INT rs, INT vs, INT m, INT mb, INT me, INT ms,
		      const planner *plnr)
{
     UNUSED(rio); UNUSED(iio); UNUSED(m);
     return (RIGHT_CPU()
	     && !NO_SIMDP(plnr)
	     && SIMD_STRIDE_OK(rs)
	     && SIMD_VSTRIDE_OK(ms)
             && (mb % VL) == 0
             && (me % VL) == 0
	     && (!d->rs || (d->rs == rs))
	     && (!d->vs || (d->vs == vs))
	     && (!d->ms || (d->ms == ms))
	  );
}

static int okp_t1f(const ct_desc *d,
		   const R *rio, const R *iio, 
		   INT rs, INT vs, INT m, INT mb, INT me, INT ms,
		   const planner *plnr)
{
     return  okp_common(d, rio, iio, rs, vs, m, mb, me, ms, plnr)
	  && iio == rio + 1
	  && ALIGNEDA(rio);
}

extern const ct_genus X(dft_t1fsimd_genus);
const ct_genus X(dft_t1fsimd_genus) = { okp_t1f, VL };

static int okp_t1fu(const ct_desc *d,
		    const R *rio, const R *iio, 
		    INT rs, INT vs, INT m, INT mb, INT me, INT ms,
		    const planner *plnr)
{
     return  okp_commonu(d, rio, iio, rs, vs, m, mb, me, ms, plnr)
	  && iio == rio + 1
	  && ALIGNED(rio);
}

extern const ct_genus X(dft_t1fusimd_genus);
const ct_genus X(dft_t1fusimd_genus) = { okp_t1fu, VL };

static int okp_t1b(const ct_desc *d,
		   const R *rio, const R *iio, 
		   INT rs, INT vs, INT m, INT mb, INT me, INT ms,
		   const planner *plnr)
{
     return  okp_common(d, rio, iio, rs, vs, m, mb, me, ms, plnr)
	  && rio == iio + 1
	  && ALIGNEDA(iio);
}

extern const ct_genus X(dft_t1bsimd_genus);
const ct_genus X(dft_t1bsimd_genus) = { okp_t1b, VL };

static int okp_t1bu(const ct_desc *d,					\
		   const R *rio, const R *iio, 				\
		   INT rs, INT vs, INT m, INT mb, INT me, INT ms,	\
		   const planner *plnr)					\
{									\
     return  okp_commonu(d, rio, iio, rs, vs, m, mb, me, ms, plnr)	\
	  && rio == iio + 1						\
	  && ALIGNED(iio);						\
}

extern const ct_genus X(dft_t1busimd_genus);
const ct_genus X(dft_t1busimd_genus) = { okp_t1bu, VL };

/* use t2* codelets only when n = m*radix is small, because
   t2* codelets use ~2n twiddle factors (instead of ~n) */
static int small_enough(const ct_desc *d, INT m)
{
     return m * d->radix <= 16384;
}

static int okp_t2f(const ct_desc *d,
		   const R *rio, const R *iio, 
		   INT rs, INT vs, INT m, INT mb, INT me, INT ms,
		   const planner *plnr)
{
     return  okp_t1f(d, rio, iio, rs, vs, m, mb, me, ms, plnr)
	  && small_enough(d, m);
}

extern const ct_genus X(dft_t2fsimd_genus);
const ct_genus X(dft_t2fsimd_genus) = { okp_t2f, VL };

static int okp_t2b(const ct_desc *d,
		   const R *rio, const R *iio, 
		   INT rs, INT vs, INT m, INT mb, INT me, INT ms,
		   const planner *plnr)
{
     return  okp_t1b(d, rio, iio, rs, vs, m, mb, me, ms, plnr)
	  && small_enough(d, m);
}

extern const ct_genus X(dft_t2bsimd_genus);
const ct_genus X(dft_t2bsimd_genus) = { okp_t2b, VL };

#endif /* HAVE_SIMD */

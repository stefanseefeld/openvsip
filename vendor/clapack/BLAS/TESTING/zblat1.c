#include "blaswrap.h"
/*  -- translated by f2c (version 19990503).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "f2c.h"

/* Common Block Declarations */

struct {
    integer icase, n, incx, incy, mode;
    logical pass;
} combla_;

#define combla_1 combla_

/* Table of constant values */

static integer c__1 = 1;
static integer c__9 = 9;
static integer c__5 = 5;
static doublereal c_b43 = 1.;

/* Main program */ MAIN__(void)
{
    /* Initialized data */

    static doublereal sfac = 9.765625e-4;

    /* Format strings */
    static char fmt_99999[] = "(\002 Complex BLAS Test Program Results\002,/"
	    "1x)";
    static char fmt_99998[] = "(\002                                    ----"
	    "- PASS -----\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    extern /* Subroutine */ int check1_(doublereal *), check2_(doublereal *);
    static integer ic;
    extern /* Subroutine */ int header_(void);

    /* Fortran I/O blocks */
    static cilist io___2 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___4 = { 0, 6, 0, fmt_99998, 0 };


/*     Test program for the COMPLEX*16 Level 1 BLAS.   
       Based upon the original BLAS test routine together with:   
       F06GAF Example Program Text */
    s_wsfe(&io___2);
    e_wsfe();
    for (ic = 1; ic <= 10; ++ic) {
	combla_1.icase = ic;
	header_();

/*        Initialize PASS, INCX, INCY, and MODE for a new case.   
          The value 9999 for INCX, INCY or MODE will appear in the   
          detailed  output, if any, for cases that do not involve   
          these parameters. */

	combla_1.pass = TRUE_;
	combla_1.incx = 9999;
	combla_1.incy = 9999;
	combla_1.mode = 9999;
	if (combla_1.icase <= 5) {
	    check2_(&sfac);
	} else if (combla_1.icase >= 6) {
	    check1_(&sfac);
	}
/*        -- Print */
	if (combla_1.pass) {
	    s_wsfe(&io___4);
	    e_wsfe();
	}
/* L20: */
    }
    s_stop("", (ftnlen)0);

    return 0;
} /* MAIN__   

   Subroutine */ int header_(void)
{
    /* Initialized data */

    static char l[6*10] = "ZDOTC " "ZDOTU " "ZAXPY " "ZCOPY " "ZSWAP " "DZNR"
	    "M2" "DZASUM" "ZSCAL " "ZDSCAL" "IZAMAX";

    /* Format strings */
    static char fmt_99999[] = "(/\002 Test of subprogram number\002,i3,12x,a"
	    "6)";

    /* Builtin functions */
    integer s_wsfe(cilist *), do_fio(integer *, char *, ftnlen), e_wsfe(void);

    /* Fortran I/O blocks */
    static cilist io___6 = { 0, 6, 0, fmt_99999, 0 };



#define l_ref(a_0,a_1) &l[(a_1)*6 + a_0 - 6]

    s_wsfe(&io___6);
    do_fio(&c__1, (char *)&combla_1.icase, (ftnlen)sizeof(integer));
    do_fio(&c__1, l_ref(0, combla_1.icase), (ftnlen)6);
    e_wsfe();
    return 0;

} /* header_ */

#undef l_ref


/* Subroutine */ int check1_(doublereal *sfac)
{
    /* Initialized data */

    static doublereal strue2[5] = { 0.,.5,.6,.7,.7 };
    static doublereal strue4[5] = { 0.,.7,1.,1.3,1.7 };
    static doublecomplex ctrue5[80]	/* was [8][5][2] */ = { {.1,.1},{1.,
	    2.},{1.,2.},{1.,2.},{1.,2.},{1.,2.},{1.,2.},{1.,2.},{-.16,-.37},{
	    3.,4.},{3.,4.},{3.,4.},{3.,4.},{3.,4.},{3.,4.},{3.,4.},{-.17,-.19}
	    ,{.13,-.39},{5.,6.},{5.,6.},{5.,6.},{5.,6.},{5.,6.},{5.,6.},{.11,
	    -.03},{-.17,.46},{-.17,-.19},{7.,8.},{7.,8.},{7.,8.},{7.,8.},{7.,
	    8.},{.19,-.17},{.32,.09},{.23,-.24},{.18,.01},{2.,3.},{2.,3.},{2.,
	    3.},{2.,3.},{.1,.1},{4.,5.},{4.,5.},{4.,5.},{4.,5.},{4.,5.},{4.,
	    5.},{4.,5.},{-.16,-.37},{6.,7.},{6.,7.},{6.,7.},{6.,7.},{6.,7.},{
	    6.,7.},{6.,7.},{-.17,-.19},{8.,9.},{.13,-.39},{2.,5.},{2.,5.},{2.,
	    5.},{2.,5.},{2.,5.},{.11,-.03},{3.,6.},{-.17,.46},{4.,7.},{-.17,
	    -.19},{7.,2.},{7.,2.},{7.,2.},{.19,-.17},{5.,8.},{.32,.09},{6.,9.}
	    ,{.23,-.24},{8.,3.},{.18,.01},{9.,4.} };
    static doublecomplex ctrue6[80]	/* was [8][5][2] */ = { {.1,.1},{1.,
	    2.},{1.,2.},{1.,2.},{1.,2.},{1.,2.},{1.,2.},{1.,2.},{.09,-.12},{
	    3.,4.},{3.,4.},{3.,4.},{3.,4.},{3.,4.},{3.,4.},{3.,4.},{.03,-.09},
	    {.15,-.03},{5.,6.},{5.,6.},{5.,6.},{5.,6.},{5.,6.},{5.,6.},{.03,
	    .03},{-.18,.03},{.03,-.09},{7.,8.},{7.,8.},{7.,8.},{7.,8.},{7.,8.}
	    ,{.09,.03},{.03,.12},{.12,.03},{.03,.06},{2.,3.},{2.,3.},{2.,3.},{
	    2.,3.},{.1,.1},{4.,5.},{4.,5.},{4.,5.},{4.,5.},{4.,5.},{4.,5.},{
	    4.,5.},{.09,-.12},{6.,7.},{6.,7.},{6.,7.},{6.,7.},{6.,7.},{6.,7.},
	    {6.,7.},{.03,-.09},{8.,9.},{.15,-.03},{2.,5.},{2.,5.},{2.,5.},{2.,
	    5.},{2.,5.},{.03,.03},{3.,6.},{-.18,.03},{4.,7.},{.03,-.09},{7.,
	    2.},{7.,2.},{7.,2.},{.09,.03},{5.,8.},{.03,.12},{6.,9.},{.12,.03},
	    {8.,3.},{.03,.06},{9.,4.} };
    static integer itrue3[5] = { 0,1,2,2,2 };
    static doublereal sa = .3;
    static doublecomplex ca = {.4,-.7};
    static doublecomplex cv[80]	/* was [8][5][2] */ = { {.1,.1},{1.,2.},{1.,
	    2.},{1.,2.},{1.,2.},{1.,2.},{1.,2.},{1.,2.},{.3,-.4},{3.,4.},{3.,
	    4.},{3.,4.},{3.,4.},{3.,4.},{3.,4.},{3.,4.},{.1,-.3},{.5,-.1},{5.,
	    6.},{5.,6.},{5.,6.},{5.,6.},{5.,6.},{5.,6.},{.1,.1},{-.6,.1},{.1,
	    -.3},{7.,8.},{7.,8.},{7.,8.},{7.,8.},{7.,8.},{.3,.1},{.1,.4},{.4,
	    .1},{.1,.2},{2.,3.},{2.,3.},{2.,3.},{2.,3.},{.1,.1},{4.,5.},{4.,
	    5.},{4.,5.},{4.,5.},{4.,5.},{4.,5.},{4.,5.},{.3,-.4},{6.,7.},{6.,
	    7.},{6.,7.},{6.,7.},{6.,7.},{6.,7.},{6.,7.},{.1,-.3},{8.,9.},{.5,
	    -.1},{2.,5.},{2.,5.},{2.,5.},{2.,5.},{2.,5.},{.1,.1},{3.,6.},{-.6,
	    .1},{4.,7.},{.1,-.3},{7.,2.},{7.,2.},{7.,2.},{.3,.1},{5.,8.},{.1,
	    .4},{6.,9.},{.4,.1},{8.,3.},{.1,.2},{9.,4.} };

    /* System generated locals */
    integer i__1, i__2, i__3;
    doublereal d__1;
    doublecomplex z__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    static integer i__;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *, 
	    doublecomplex *, integer *), ctest_(integer *, doublecomplex *, 
	    doublecomplex *, doublecomplex *, doublereal *);
    static doublecomplex mwpcs[5], mwpct[5];
    extern /* Subroutine */ int itest1_(integer *, integer *);
    extern doublereal dznrm2_(integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int stest1_(doublereal *, doublereal *, 
	    doublereal *, doublereal *);
    static doublecomplex cx[8];
    extern /* Subroutine */ int zdscal_(integer *, doublereal *, 
	    doublecomplex *, integer *);
    extern integer izamax_(integer *, doublecomplex *, integer *);
    extern doublereal dzasum_(integer *, doublecomplex *, integer *);
    static integer np1, len;

    /* Fortran I/O blocks */
    static cilist io___19 = { 0, 6, 0, 0, 0 };



#define ctrue5_subscr(a_1,a_2,a_3) ((a_3)*5 + (a_2))*8 + a_1 - 49
#define ctrue5_ref(a_1,a_2,a_3) ctrue5[ctrue5_subscr(a_1,a_2,a_3)]
#define ctrue6_subscr(a_1,a_2,a_3) ((a_3)*5 + (a_2))*8 + a_1 - 49
#define ctrue6_ref(a_1,a_2,a_3) ctrue6[ctrue6_subscr(a_1,a_2,a_3)]
#define cv_subscr(a_1,a_2,a_3) ((a_3)*5 + (a_2))*8 + a_1 - 49
#define cv_ref(a_1,a_2,a_3) cv[cv_subscr(a_1,a_2,a_3)]

    for (combla_1.incx = 1; combla_1.incx <= 2; ++combla_1.incx) {
	for (np1 = 1; np1 <= 5; ++np1) {
	    combla_1.n = np1 - 1;
	    len = max(combla_1.n,1) << 1;
	    i__1 = len;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__ - 1;
		i__3 = cv_subscr(i__, np1, combla_1.incx);
		cx[i__2].r = cv[i__3].r, cx[i__2].i = cv[i__3].i;
/* L20: */
	    }
	    if (combla_1.icase == 6) {
		d__1 = dznrm2_(&combla_1.n, cx, &combla_1.incx);
		stest1_(&d__1, &strue2[np1 - 1], &strue2[np1 - 1], sfac);
	    } else if (combla_1.icase == 7) {
		d__1 = dzasum_(&combla_1.n, cx, &combla_1.incx);
		stest1_(&d__1, &strue4[np1 - 1], &strue4[np1 - 1], sfac);
	    } else if (combla_1.icase == 8) {
		zscal_(&combla_1.n, &ca, cx, &combla_1.incx);
		ctest_(&len, cx, &ctrue5_ref(1, np1, combla_1.incx), &
			ctrue5_ref(1, np1, combla_1.incx), sfac);
	    } else if (combla_1.icase == 9) {
		zdscal_(&combla_1.n, &sa, cx, &combla_1.incx);
		ctest_(&len, cx, &ctrue6_ref(1, np1, combla_1.incx), &
			ctrue6_ref(1, np1, combla_1.incx), sfac);
	    } else if (combla_1.icase == 10) {
		i__1 = izamax_(&combla_1.n, cx, &combla_1.incx);
		itest1_(&i__1, &itrue3[np1 - 1]);
	    } else {
		s_wsle(&io___19);
		do_lio(&c__9, &c__1, " Shouldn't be here in CHECK1", (ftnlen)
			28);
		e_wsle();
		s_stop("", (ftnlen)0);
	    }

/* L40: */
	}
/* L60: */
    }

    combla_1.incx = 1;
    if (combla_1.icase == 8) {
/*        ZSCAL   
          Add a test for alpha equal to zero. */
	ca.r = 0., ca.i = 0.;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    mwpct[i__1].r = 0., mwpct[i__1].i = 0.;
	    i__1 = i__ - 1;
	    mwpcs[i__1].r = 1., mwpcs[i__1].i = 1.;
/* L80: */
	}
	zscal_(&c__5, &ca, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
    } else if (combla_1.icase == 9) {
/*        ZDSCAL   
          Add a test for alpha equal to zero. */
	sa = 0.;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    mwpct[i__1].r = 0., mwpct[i__1].i = 0.;
	    i__1 = i__ - 1;
	    mwpcs[i__1].r = 1., mwpcs[i__1].i = 1.;
/* L100: */
	}
	zdscal_(&c__5, &sa, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
/*        Add a test for alpha equal to one. */
	sa = 1.;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    mwpct[i__1].r = cx[i__2].r, mwpct[i__1].i = cx[i__2].i;
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    mwpcs[i__1].r = cx[i__2].r, mwpcs[i__1].i = cx[i__2].i;
/* L120: */
	}
	zdscal_(&c__5, &sa, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
/*        Add a test for alpha equal to minus one. */
	sa = -1.;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    z__1.r = -cx[i__2].r, z__1.i = -cx[i__2].i;
	    mwpct[i__1].r = z__1.r, mwpct[i__1].i = z__1.i;
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    z__1.r = -cx[i__2].r, z__1.i = -cx[i__2].i;
	    mwpcs[i__1].r = z__1.r, mwpcs[i__1].i = z__1.i;
/* L140: */
	}
	zdscal_(&c__5, &sa, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
    }
    return 0;
} /* check1_ */

#undef cv_ref
#undef cv_subscr
#undef ctrue6_ref
#undef ctrue6_subscr
#undef ctrue5_ref
#undef ctrue5_subscr


/* Subroutine */ int check2_(doublereal *sfac)
{
    /* Initialized data */

    static doublecomplex ca = {.4,-.7};
    static integer incxs[4] = { 1,2,-2,-1 };
    static integer incys[4] = { 1,-2,1,-2 };
    static integer lens[8]	/* was [4][2] */ = { 1,1,2,4,1,1,3,7 };
    static integer ns[4] = { 0,1,2,4 };
    static doublecomplex cx1[7] = { {.7,-.8},{-.4,-.7},{-.1,-.9},{.2,-.8},{
	    -.9,-.4},{.1,.4},{-.6,.6} };
    static doublecomplex cy1[7] = { {.6,-.6},{-.9,.5},{.7,-.6},{.1,-.5},{-.1,
	    -.2},{-.5,-.3},{.8,-.7} };
    static doublecomplex ct8[112]	/* was [7][4][4] */ = { {.6,-.6},{0.,
	    0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.32,-1.41},{0.,0.},{
	    0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.32,-1.41},{-1.55,.5},{0.,
	    0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.32,-1.41},{-1.55,.5},{.03,
	    -.89},{-.38,-.96},{0.,0.},{0.,0.},{0.,0.},{.6,-.6},{0.,0.},{0.,0.}
	    ,{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.32,-1.41},{0.,0.},{0.,0.},{0.,
	    0.},{0.,0.},{0.,0.},{0.,0.},{-.07,-.89},{-.9,.5},{.42,-1.41},{0.,
	    0.},{0.,0.},{0.,0.},{0.,0.},{.78,.06},{-.9,.5},{.06,-.13},{.1,-.5}
	    ,{-.77,-.49},{-.5,-.3},{.52,-1.51},{.6,-.6},{0.,0.},{0.,0.},{0.,
	    0.},{0.,0.},{0.,0.},{0.,0.},{.32,-1.41},{0.,0.},{0.,0.},{0.,0.},{
	    0.,0.},{0.,0.},{0.,0.},{-.07,-.89},{-1.18,-.31},{0.,0.},{0.,0.},{
	    0.,0.},{0.,0.},{0.,0.},{.78,.06},{-1.54,.97},{.03,-.89},{-.18,
	    -1.31},{0.,0.},{0.,0.},{0.,0.},{.6,-.6},{0.,0.},{0.,0.},{0.,0.},{
	    0.,0.},{0.,0.},{0.,0.},{.32,-1.41},{0.,0.},{0.,0.},{0.,0.},{0.,0.}
	    ,{0.,0.},{0.,0.},{.32,-1.41},{-.9,.5},{.05,-.6},{0.,0.},{0.,0.},{
	    0.,0.},{0.,0.},{.32,-1.41},{-.9,.5},{.05,-.6},{.1,-.5},{-.77,-.49}
	    ,{-.5,-.3},{.32,-1.16} };
    static doublecomplex ct7[16]	/* was [4][4] */ = { {0.,0.},{-.06,
	    -.9},{.65,-.47},{-.34,-1.22},{0.,0.},{-.06,-.9},{-.59,-1.46},{
	    -1.04,-.04},{0.,0.},{-.06,-.9},{-.83,.59},{.07,-.37},{0.,0.},{
	    -.06,-.9},{-.76,-1.15},{-1.33,-1.82} };
    static doublecomplex ct6[16]	/* was [4][4] */ = { {0.,0.},{.9,.06},
	    {.91,-.77},{1.8,-.1},{0.,0.},{.9,.06},{1.45,.74},{.2,.9},{0.,0.},{
	    .9,.06},{-.55,.23},{.83,-.39},{0.,0.},{.9,.06},{1.04,.79},{1.95,
	    1.22} };
    static doublecomplex ct10x[112]	/* was [7][4][4] */ = { {.7,-.8},{0.,
	    0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.6,-.6},{0.,0.},{0.,
	    0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.6,-.6},{-.9,.5},{0.,0.},{0.,
	    0.},{0.,0.},{0.,0.},{0.,0.},{.6,-.6},{-.9,.5},{.7,-.6},{.1,-.5},{
	    0.,0.},{0.,0.},{0.,0.},{.7,-.8},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{
	    0.,0.},{0.,0.},{.6,-.6},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{
	    0.,0.},{.7,-.6},{-.4,-.7},{.6,-.6},{0.,0.},{0.,0.},{0.,0.},{0.,0.}
	    ,{.8,-.7},{-.4,-.7},{-.1,-.2},{.2,-.8},{.7,-.6},{.1,.4},{.6,-.6},{
	    .7,-.8},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.6,-.6},{
	    0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{-.9,.5},{-.4,-.7},
	    {.6,-.6},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.1,-.5},{-.4,-.7},{.7,
	    -.6},{.2,-.8},{-.9,.5},{.1,.4},{.6,-.6},{.7,-.8},{0.,0.},{0.,0.},{
	    0.,0.},{0.,0.},{0.,0.},{0.,0.},{.6,-.6},{0.,0.},{0.,0.},{0.,0.},{
	    0.,0.},{0.,0.},{0.,0.},{.6,-.6},{.7,-.6},{0.,0.},{0.,0.},{0.,0.},{
	    0.,0.},{0.,0.},{.6,-.6},{.7,-.6},{-.1,-.2},{.8,-.7},{0.,0.},{0.,
	    0.},{0.,0.} };
    static doublecomplex ct10y[112]	/* was [7][4][4] */ = { {.6,-.6},{0.,
	    0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.7,-.8},{0.,0.},{0.,
	    0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.7,-.8},{-.4,-.7},{0.,0.},{
	    0.,0.},{0.,0.},{0.,0.},{0.,0.},{.7,-.8},{-.4,-.7},{-.1,-.9},{.2,
	    -.8},{0.,0.},{0.,0.},{0.,0.},{.6,-.6},{0.,0.},{0.,0.},{0.,0.},{0.,
	    0.},{0.,0.},{0.,0.},{.7,-.8},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,
	    0.},{0.,0.},{-.1,-.9},{-.9,.5},{.7,-.8},{0.,0.},{0.,0.},{0.,0.},{
	    0.,0.},{-.6,.6},{-.9,.5},{-.9,-.4},{.1,-.5},{-.1,-.9},{-.5,-.3},{
	    .7,-.8},{.6,-.6},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{
	    .7,-.8},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{-.1,-.9},
	    {.7,-.8},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{-.6,.6},{-.9,
	    -.4},{-.1,-.9},{.7,-.8},{0.,0.},{0.,0.},{0.,0.},{.6,-.6},{0.,0.},{
	    0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{.7,-.8},{0.,0.},{0.,0.},{
	    0.,0.},{0.,0.},{0.,0.},{0.,0.},{.7,-.8},{-.9,.5},{-.4,-.7},{0.,0.}
	    ,{0.,0.},{0.,0.},{0.,0.},{.7,-.8},{-.9,.5},{-.4,-.7},{.1,-.5},{
	    -.1,-.9},{-.5,-.3},{.2,-.8} };
    static doublecomplex csize1[4] = { {0.,0.},{.9,.9},{1.63,1.73},{2.9,2.78} 
	    };
    static doublecomplex csize3[14] = { {0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,
	    0.},{0.,0.},{0.,0.},{1.17,1.17},{1.17,1.17},{1.17,1.17},{1.17,
	    1.17},{1.17,1.17},{1.17,1.17},{1.17,1.17} };
    static doublecomplex csize2[14]	/* was [7][2] */ = { {0.,0.},{0.,0.},{
	    0.,0.},{0.,0.},{0.,0.},{0.,0.},{0.,0.},{1.54,1.54},{1.54,1.54},{
	    1.54,1.54},{1.54,1.54},{1.54,1.54},{1.54,1.54},{1.54,1.54} };

    /* System generated locals */
    integer i__1, i__2;
    doublecomplex z__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    static doublecomplex cdot[1];
    static integer lenx, leny, i__;
    extern /* Subroutine */ int ctest_(integer *, doublecomplex *, 
	    doublecomplex *, doublecomplex *, doublereal *);
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    static integer ksize;
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *);
    extern /* Double Complex */ VOID zdotu_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int zswap_(integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *), zaxpy_(integer *, doublecomplex *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    static integer ki, kn;
    static doublecomplex cx[7], cy[7];
    static integer mx, my;

    /* Fortran I/O blocks */
    static cilist io___48 = { 0, 6, 0, 0, 0 };



#define ct10x_subscr(a_1,a_2,a_3) ((a_3)*4 + (a_2))*7 + a_1 - 36
#define ct10x_ref(a_1,a_2,a_3) ct10x[ct10x_subscr(a_1,a_2,a_3)]
#define ct10y_subscr(a_1,a_2,a_3) ((a_3)*4 + (a_2))*7 + a_1 - 36
#define ct10y_ref(a_1,a_2,a_3) ct10y[ct10y_subscr(a_1,a_2,a_3)]
#define lens_ref(a_1,a_2) lens[(a_2)*4 + a_1 - 5]
#define csize2_subscr(a_1,a_2) (a_2)*7 + a_1 - 8
#define csize2_ref(a_1,a_2) csize2[csize2_subscr(a_1,a_2)]
#define ct6_subscr(a_1,a_2) (a_2)*4 + a_1 - 5
#define ct6_ref(a_1,a_2) ct6[ct6_subscr(a_1,a_2)]
#define ct7_subscr(a_1,a_2) (a_2)*4 + a_1 - 5
#define ct7_ref(a_1,a_2) ct7[ct7_subscr(a_1,a_2)]
#define ct8_subscr(a_1,a_2,a_3) ((a_3)*4 + (a_2))*7 + a_1 - 36
#define ct8_ref(a_1,a_2,a_3) ct8[ct8_subscr(a_1,a_2,a_3)]

    for (ki = 1; ki <= 4; ++ki) {
	combla_1.incx = incxs[ki - 1];
	combla_1.incy = incys[ki - 1];
	mx = abs(combla_1.incx);
	my = abs(combla_1.incy);

	for (kn = 1; kn <= 4; ++kn) {
	    combla_1.n = ns[kn - 1];
	    ksize = min(2,kn);
	    lenx = lens_ref(kn, mx);
	    leny = lens_ref(kn, my);
	    for (i__ = 1; i__ <= 7; ++i__) {
		i__1 = i__ - 1;
		i__2 = i__ - 1;
		cx[i__1].r = cx1[i__2].r, cx[i__1].i = cx1[i__2].i;
		i__1 = i__ - 1;
		i__2 = i__ - 1;
		cy[i__1].r = cy1[i__2].r, cy[i__1].i = cy1[i__2].i;
/* L20: */
	    }
	    if (combla_1.icase == 1) {
		zdotc_(&z__1, &combla_1.n, cx, &combla_1.incx, cy, &
			combla_1.incy);
		cdot[0].r = z__1.r, cdot[0].i = z__1.i;
		ctest_(&c__1, cdot, &ct6_ref(kn, ki), &csize1[kn - 1], sfac);
	    } else if (combla_1.icase == 2) {
		zdotu_(&z__1, &combla_1.n, cx, &combla_1.incx, cy, &
			combla_1.incy);
		cdot[0].r = z__1.r, cdot[0].i = z__1.i;
		ctest_(&c__1, cdot, &ct7_ref(kn, ki), &csize1[kn - 1], sfac);
	    } else if (combla_1.icase == 3) {
		zaxpy_(&combla_1.n, &ca, cx, &combla_1.incx, cy, &
			combla_1.incy);
		ctest_(&leny, cy, &ct8_ref(1, kn, ki), &csize2_ref(1, ksize), 
			sfac);
	    } else if (combla_1.icase == 4) {
		zcopy_(&combla_1.n, cx, &combla_1.incx, cy, &combla_1.incy);
		ctest_(&leny, cy, &ct10y_ref(1, kn, ki), csize3, &c_b43);
	    } else if (combla_1.icase == 5) {
		zswap_(&combla_1.n, cx, &combla_1.incx, cy, &combla_1.incy);
		ctest_(&lenx, cx, &ct10x_ref(1, kn, ki), csize3, &c_b43);
		ctest_(&leny, cy, &ct10y_ref(1, kn, ki), csize3, &c_b43);
	    } else {
		s_wsle(&io___48);
		do_lio(&c__9, &c__1, " Shouldn't be here in CHECK2", (ftnlen)
			28);
		e_wsle();
		s_stop("", (ftnlen)0);
	    }

/* L40: */
	}
/* L60: */
    }
    return 0;
} /* check2_ */

#undef ct8_ref
#undef ct8_subscr
#undef ct7_ref
#undef ct7_subscr
#undef ct6_ref
#undef ct6_subscr
#undef csize2_ref
#undef csize2_subscr
#undef lens_ref
#undef ct10y_ref
#undef ct10y_subscr
#undef ct10x_ref
#undef ct10x_subscr


/* Subroutine */ int stest_(integer *len, doublereal *scomp, doublereal *
	strue, doublereal *ssize, doublereal *sfac)
{
    /* Format strings */
    static char fmt_99999[] = "(\002                                       F"
	    "AIL\002)";
    static char fmt_99998[] = "(/\002 CASE  N INCX INCY MODE  I             "
	    "               \002,\002 COMP(I)                             TRU"
	    "E(I)  DIFFERENCE\002,\002     SIZE(I)\002,/1x)";
    static char fmt_99997[] = "(1x,i4,i3,3i5,i3,2d36.8,2d12.4)";

    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2, d__3, d__4, d__5;

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    static integer i__;
    extern doublereal sdiff_(doublereal *, doublereal *);
    static doublereal sd;

    /* Fortran I/O blocks */
    static cilist io___51 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___52 = { 0, 6, 0, fmt_99998, 0 };
    static cilist io___53 = { 0, 6, 0, fmt_99997, 0 };


/*     ********************************* STEST **************************   

       THIS SUBR COMPARES ARRAYS  SCOMP() AND STRUE() OF LENGTH LEN TO   
       SEE IF THE TERM BY TERM DIFFERENCES, MULTIPLIED BY SFAC, ARE   
       NEGLIGIBLE.   

       C. L. LAWSON, JPL, 1974 DEC 10   


       Parameter adjustments */
    --ssize;
    --strue;
    --scomp;

    /* Function Body */
    i__1 = *len;
    for (i__ = 1; i__ <= i__1; ++i__) {
	sd = scomp[i__] - strue[i__];
	d__4 = (d__1 = ssize[i__], abs(d__1)) + (d__2 = *sfac * sd, abs(d__2))
		;
	d__5 = (d__3 = ssize[i__], abs(d__3));
	if (sdiff_(&d__4, &d__5) == 0.) {
	    goto L40;
	}

/*                             HERE    SCOMP(I) IS NOT CLOSE TO STRUE(I). */

	if (! combla_1.pass) {
	    goto L20;
	}
/*                             PRINT FAIL MESSAGE AND HEADER. */
	combla_1.pass = FALSE_;
	s_wsfe(&io___51);
	e_wsfe();
	s_wsfe(&io___52);
	e_wsfe();
L20:
	s_wsfe(&io___53);
	do_fio(&c__1, (char *)&combla_1.icase, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.n, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.incx, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.incy, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&combla_1.mode, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&i__, (ftnlen)sizeof(integer));
	do_fio(&c__1, (char *)&scomp[i__], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&strue[i__], (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&sd, (ftnlen)sizeof(doublereal));
	do_fio(&c__1, (char *)&ssize[i__], (ftnlen)sizeof(doublereal));
	e_wsfe();
L40:
	;
    }
    return 0;

} /* stest_   

   Subroutine */ int stest1_(doublereal *scomp1, doublereal *strue1, 
	doublereal *ssize, doublereal *sfac)
{
    static doublereal scomp[1], strue[1];
    extern /* Subroutine */ int stest_(integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *);

/*     ************************* STEST1 *****************************   

       THIS IS AN INTERFACE SUBROUTINE TO ACCOMODATE THE FORTRAN   
       REQUIREMENT THAT WHEN A DUMMY ARGUMENT IS AN ARRAY, THE   
       ACTUAL ARGUMENT MUST ALSO BE AN ARRAY OR AN ARRAY ELEMENT.   

       C.L. LAWSON, JPL, 1978 DEC 6   


       Parameter adjustments */
    --ssize;

    /* Function Body */
    scomp[0] = *scomp1;
    strue[0] = *strue1;
    stest_(&c__1, scomp, strue, &ssize[1], sfac);

    return 0;
} /* stest1_ */

doublereal sdiff_(doublereal *sa, doublereal *sb)
{
    /* System generated locals */
    doublereal ret_val;

/*     ********************************* SDIFF **************************   
       COMPUTES DIFFERENCE OF TWO NUMBERS.  C. L. LAWSON, JPL 1974 FEB 15 */

    ret_val = *sa - *sb;
    return ret_val;
} /* sdiff_   

   Subroutine */ int ctest_(integer *len, doublecomplex *ccomp, doublecomplex 
	*ctrue, doublecomplex *csize, doublereal *sfac)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    static integer i__;
    static doublereal scomp[20], ssize[20], strue[20];
    extern /* Subroutine */ int stest_(integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *);

/*     **************************** CTEST *****************************   

       C.L. LAWSON, JPL, 1978 DEC 6   

       Parameter adjustments */
    --csize;
    --ctrue;
    --ccomp;

    /* Function Body */
    i__1 = *len;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	scomp[(i__ << 1) - 2] = ccomp[i__2].r;
	scomp[(i__ << 1) - 1] = d_imag(&ccomp[i__]);
	i__2 = i__;
	strue[(i__ << 1) - 2] = ctrue[i__2].r;
	strue[(i__ << 1) - 1] = d_imag(&ctrue[i__]);
	i__2 = i__;
	ssize[(i__ << 1) - 2] = csize[i__2].r;
	ssize[(i__ << 1) - 1] = d_imag(&csize[i__]);
/* L20: */
    }

    i__1 = *len << 1;
    stest_(&i__1, scomp, strue, ssize, sfac);
    return 0;
} /* ctest_   

   Subroutine */ int itest1_(integer *icomp, integer *itrue)
{
    /* Format strings */
    static char fmt_99999[] = "(\002                                       F"
	    "AIL\002)";
    static char fmt_99998[] = "(/\002 CASE  N INCX INCY MODE                "
	    "               \002,\002 COMP                                TRU"
	    "E     DIFFERENCE\002,/1x)";
    static char fmt_99997[] = "(1x,i4,i3,3i5,2i36,i12)";

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    static integer id;

    /* Fortran I/O blocks */
    static cilist io___60 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___61 = { 0, 6, 0, fmt_99998, 0 };
    static cilist io___63 = { 0, 6, 0, fmt_99997, 0 };


/*     ********************************* ITEST1 *************************   

       THIS SUBROUTINE COMPARES THE VARIABLES ICOMP AND ITRUE FOR   
       EQUALITY.   
       C. L. LAWSON, JPL, 1974 DEC 10 */

    if (*icomp == *itrue) {
	goto L40;
    }

/*                            HERE ICOMP IS NOT EQUAL TO ITRUE. */

    if (! combla_1.pass) {
	goto L20;
    }
/*                             PRINT FAIL MESSAGE AND HEADER. */
    combla_1.pass = FALSE_;
    s_wsfe(&io___60);
    e_wsfe();
    s_wsfe(&io___61);
    e_wsfe();
L20:
    id = *icomp - *itrue;
    s_wsfe(&io___63);
    do_fio(&c__1, (char *)&combla_1.icase, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.n, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.incx, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.incy, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&combla_1.mode, (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&(*icomp), (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&(*itrue), (ftnlen)sizeof(integer));
    do_fio(&c__1, (char *)&id, (ftnlen)sizeof(integer));
    e_wsfe();
L40:
    return 0;

} /* itest1_   

   Main program alias */ int zblat1_ () { MAIN__ (); return 0; }

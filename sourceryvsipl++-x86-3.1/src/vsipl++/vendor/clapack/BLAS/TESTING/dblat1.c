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
static doublereal c_b34 = 1.;
static integer c__5 = 5;

/* Main program */ MAIN__(void)
{
    /* Initialized data */

    static doublereal sfac = 9.765625e-4;

    /* Format strings */
    static char fmt_99999[] = "(\002 Real BLAS Test Program Results\002,/1x)";
    static char fmt_99998[] = "(\002                                    ----"
	    "- PASS -----\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    extern /* Subroutine */ int check0_(doublereal *), check1_(doublereal *), 
	    check2_(doublereal *), check3_(doublereal *);
    static integer ic;
    extern /* Subroutine */ int header_(void);

    /* Fortran I/O blocks */
    static cilist io___2 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___4 = { 0, 6, 0, fmt_99998, 0 };


/*     Test program for the DOUBLE PRECISION Level 1 BLAS.   
       Based upon the original BLAS test routine together with:   
       F06EAF Example Program Text */
    s_wsfe(&io___2);
    e_wsfe();
    for (ic = 1; ic <= 10; ++ic) {
	combla_1.icase = ic;
	header_();


	combla_1.pass = TRUE_;
	combla_1.incx = 9999;
	combla_1.incy = 9999;
	combla_1.mode = 9999;
	if (combla_1.icase == 3) {
	    check0_(&sfac);
	} else if (combla_1.icase == 7 || combla_1.icase == 8 || 
		combla_1.icase == 9 || combla_1.icase == 10) {
	    check1_(&sfac);
	} else if (combla_1.icase == 1 || combla_1.icase == 2 || 
		combla_1.icase == 5 || combla_1.icase == 6) {
	    check2_(&sfac);
	} else if (combla_1.icase == 4) {
	    check3_(&sfac);
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

    static char l[6*10] = " DDOT " "DAXPY " "DROTG " " DROT " "DCOPY " "DSWA"
	    "P " "DNRM2 " "DASUM " "DSCAL " "IDAMAX";

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


/* Subroutine */ int check0_(doublereal *sfac)
{
    /* Initialized data */

    static doublereal ds1[8] = { .8,.6,.8,-.6,.8,0.,1.,0. };
    static doublereal datrue[8] = { .5,.5,.5,-.5,-.5,0.,1.,1. };
    static doublereal dbtrue[8] = { 0.,.6,0.,-.6,0.,0.,1.,0. };
    static doublereal da1[8] = { .3,.4,-.3,-.4,-.3,0.,0.,1. };
    static doublereal db1[8] = { .4,.3,.4,.3,-.4,0.,1.,0. };
    static doublereal dc1[8] = { .6,.8,-.6,.8,.6,1.,0.,1. };

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    static integer k;
    extern /* Subroutine */ int drotg_(doublereal *, doublereal *, doublereal 
	    *, doublereal *), stest1_(doublereal *, doublereal *, doublereal *
	    , doublereal *);
    static doublereal sa, sb, sc, ss;

    /* Fortran I/O blocks */
    static cilist io___19 = { 0, 6, 0, 0, 0 };



/*     Compute true values which cannot be prestored   
       in decimal notation */

    dbtrue[0] = 1.6666666666666667;
    dbtrue[2] = -1.6666666666666667;
    dbtrue[4] = 1.6666666666666667;

    for (k = 1; k <= 8; ++k) {
	combla_1.n = k;
	if (combla_1.icase == 3) {
	    if (k > 8) {
		goto L40;
	    }
	    sa = da1[k - 1];
	    sb = db1[k - 1];
	    drotg_(&sa, &sb, &sc, &ss);
	    stest1_(&sa, &datrue[k - 1], &datrue[k - 1], sfac);
	    stest1_(&sb, &dbtrue[k - 1], &dbtrue[k - 1], sfac);
	    stest1_(&sc, &dc1[k - 1], &dc1[k - 1], sfac);
	    stest1_(&ss, &ds1[k - 1], &ds1[k - 1], sfac);
	} else {
	    s_wsle(&io___19);
	    do_lio(&c__9, &c__1, " Shouldn't be here in CHECK0", (ftnlen)28);
	    e_wsle();
	    s_stop("", (ftnlen)0);
	}
/* L20: */
    }
L40:
    return 0;
} /* check0_   

   Subroutine */ int check1_(doublereal *sfac)
{
    /* Initialized data */

    static doublereal sa[10] = { .3,-1.,0.,1.,.3,.3,.3,.3,.3,.3 };
    static doublereal dv[80]	/* was [8][5][2] */ = { .1,2.,2.,2.,2.,2.,2.,
	    2.,.3,3.,3.,3.,3.,3.,3.,3.,.3,-.4,4.,4.,4.,4.,4.,4.,.2,-.6,.3,5.,
	    5.,5.,5.,5.,.1,-.3,.5,-.1,6.,6.,6.,6.,.1,8.,8.,8.,8.,8.,8.,8.,.3,
	    9.,9.,9.,9.,9.,9.,9.,.3,2.,-.4,2.,2.,2.,2.,2.,.2,3.,-.6,5.,.3,2.,
	    2.,2.,.1,4.,-.3,6.,-.5,7.,-.1,3. };
    static doublereal dtrue1[5] = { 0.,.3,.5,.7,.6 };
    static doublereal dtrue3[5] = { 0.,.3,.7,1.1,1. };
    static doublereal dtrue5[80]	/* was [8][5][2] */ = { .1,2.,2.,2.,
	    2.,2.,2.,2.,-.3,3.,3.,3.,3.,3.,3.,3.,0.,0.,4.,4.,4.,4.,4.,4.,.2,
	    -.6,.3,5.,5.,5.,5.,5.,.03,-.09,.15,-.03,6.,6.,6.,6.,.1,8.,8.,8.,
	    8.,8.,8.,8.,.09,9.,9.,9.,9.,9.,9.,9.,.09,2.,-.12,2.,2.,2.,2.,2.,
	    .06,3.,-.18,5.,.09,2.,2.,2.,.03,4.,-.09,6.,-.15,7.,-.03,3. };
    static integer itrue2[5] = { 0,1,2,2,3 };

    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    extern doublereal dnrm2_(integer *, doublereal *, integer *);
    static integer i__;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    extern doublereal dasum_(integer *, doublereal *, integer *);
    static doublereal stemp[1], strue[8];
    extern /* Subroutine */ int stest_(integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *), itest1_(integer *, integer *), 
	    stest1_(doublereal *, doublereal *, doublereal *, doublereal *);
    extern integer idamax_(integer *, doublereal *, integer *);
    static doublereal sx[8];
    static integer np1, len;

    /* Fortran I/O blocks */
    static cilist io___32 = { 0, 6, 0, 0, 0 };



#define dtrue5_ref(a_1,a_2,a_3) dtrue5[((a_3)*5 + (a_2))*8 + a_1 - 49]
#define dv_ref(a_1,a_2,a_3) dv[((a_3)*5 + (a_2))*8 + a_1 - 49]

    for (combla_1.incx = 1; combla_1.incx <= 2; ++combla_1.incx) {
	for (np1 = 1; np1 <= 5; ++np1) {
	    combla_1.n = np1 - 1;
	    len = max(combla_1.n,1) << 1;
	    i__1 = len;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		sx[i__ - 1] = dv_ref(i__, np1, combla_1.incx);
/* L20: */
	    }

	    if (combla_1.icase == 7) {
		stemp[0] = dtrue1[np1 - 1];
		d__1 = dnrm2_(&combla_1.n, sx, &combla_1.incx);
		stest1_(&d__1, stemp, stemp, sfac);
	    } else if (combla_1.icase == 8) {
		stemp[0] = dtrue3[np1 - 1];
		d__1 = dasum_(&combla_1.n, sx, &combla_1.incx);
		stest1_(&d__1, stemp, stemp, sfac);
	    } else if (combla_1.icase == 9) {
		dscal_(&combla_1.n, &sa[(combla_1.incx - 1) * 5 + np1 - 1], 
			sx, &combla_1.incx);
		i__1 = len;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    strue[i__ - 1] = dtrue5_ref(i__, np1, combla_1.incx);
/* L40: */
		}
		stest_(&len, sx, strue, strue, sfac);
	    } else if (combla_1.icase == 10) {
		i__1 = idamax_(&combla_1.n, sx, &combla_1.incx);
		itest1_(&i__1, &itrue2[np1 - 1]);
	    } else {
		s_wsle(&io___32);
		do_lio(&c__9, &c__1, " Shouldn't be here in CHECK1", (ftnlen)
			28);
		e_wsle();
		s_stop("", (ftnlen)0);
	    }
/* L60: */
	}
/* L80: */
    }
    return 0;
} /* check1_ */

#undef dv_ref
#undef dtrue5_ref


/* Subroutine */ int check2_(doublereal *sfac)
{
    /* Initialized data */

    static doublereal sa = .3;
    static integer incxs[4] = { 1,2,-2,-1 };
    static integer incys[4] = { 1,-2,1,-2 };
    static integer lens[8]	/* was [4][2] */ = { 1,1,2,4,1,1,3,7 };
    static integer ns[4] = { 0,1,2,4 };
    static doublereal dx1[7] = { .6,.1,-.5,.8,.9,-.3,-.4 };
    static doublereal dy1[7] = { .5,-.9,.3,.7,-.6,.2,.8 };
    static doublereal dt7[16]	/* was [4][4] */ = { 0.,.3,.21,.62,0.,.3,-.07,
	    .85,0.,.3,-.79,-.74,0.,.3,.33,1.27 };
    static doublereal dt8[112]	/* was [7][4][4] */ = { .5,0.,0.,0.,0.,0.,0.,
	    .68,0.,0.,0.,0.,0.,0.,.68,-.87,0.,0.,0.,0.,0.,.68,-.87,.15,.94,0.,
	    0.,0.,.5,0.,0.,0.,0.,0.,0.,.68,0.,0.,0.,0.,0.,0.,.35,-.9,.48,0.,
	    0.,0.,0.,.38,-.9,.57,.7,-.75,.2,.98,.5,0.,0.,0.,0.,0.,0.,.68,0.,
	    0.,0.,0.,0.,0.,.35,-.72,0.,0.,0.,0.,0.,.38,-.63,.15,.88,0.,0.,0.,
	    .5,0.,0.,0.,0.,0.,0.,.68,0.,0.,0.,0.,0.,0.,.68,-.9,.33,0.,0.,0.,
	    0.,.68,-.9,.33,.7,-.75,.2,1.04 };
    static doublereal dt10x[112]	/* was [7][4][4] */ = { .6,0.,0.,0.,
	    0.,0.,0.,.5,0.,0.,0.,0.,0.,0.,.5,-.9,0.,0.,0.,0.,0.,.5,-.9,.3,.7,
	    0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.5,0.,0.,0.,0.,0.,0.,.3,.1,.5,0.,0.,
	    0.,0.,.8,.1,-.6,.8,.3,-.3,.5,.6,0.,0.,0.,0.,0.,0.,.5,0.,0.,0.,0.,
	    0.,0.,-.9,.1,.5,0.,0.,0.,0.,.7,.1,.3,.8,-.9,-.3,.5,.6,0.,0.,0.,0.,
	    0.,0.,.5,0.,0.,0.,0.,0.,0.,.5,.3,0.,0.,0.,0.,0.,.5,.3,-.6,.8,0.,
	    0.,0. };
    static doublereal dt10y[112]	/* was [7][4][4] */ = { .5,0.,0.,0.,
	    0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.6,.1,0.,0.,0.,0.,0.,.6,.1,-.5,.8,
	    0.,0.,0.,.5,0.,0.,0.,0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,-.5,-.9,.6,0.,
	    0.,0.,0.,-.4,-.9,.9,.7,-.5,.2,.6,.5,0.,0.,0.,0.,0.,0.,.6,0.,0.,0.,
	    0.,0.,0.,-.5,.6,0.,0.,0.,0.,0.,-.4,.9,-.5,.6,0.,0.,0.,.5,0.,0.,0.,
	    0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.6,-.9,.1,0.,0.,0.,0.,.6,-.9,.1,.7,
	    -.5,.2,.8 };
    static doublereal ssize1[4] = { 0.,.3,1.6,3.2 };
    static doublereal ssize2[28]	/* was [14][2] */ = { 0.,0.,0.,0.,0.,
	    0.,0.,0.,0.,0.,0.,0.,0.,0.,1.17,1.17,1.17,1.17,1.17,1.17,1.17,
	    1.17,1.17,1.17,1.17,1.17,1.17,1.17 };

    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    static integer lenx, leny, i__, j;
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *), dswap_(integer *, doublereal *, integer 
	    *, doublereal *, integer *);
    static integer ksize;
    extern /* Subroutine */ int daxpy_(integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, integer *), stest_(integer *, doublereal 
	    *, doublereal *, doublereal *, doublereal *), stest1_(doublereal *
	    , doublereal *, doublereal *, doublereal *);
    static integer ki, kn, mx, my;
    static doublereal sx[7], sy[7], stx[7], sty[7];

    /* Fortran I/O blocks */
    static cilist io___63 = { 0, 6, 0, 0, 0 };



#define dt10x_ref(a_1,a_2,a_3) dt10x[((a_3)*4 + (a_2))*7 + a_1 - 36]
#define dt10y_ref(a_1,a_2,a_3) dt10y[((a_3)*4 + (a_2))*7 + a_1 - 36]
#define lens_ref(a_1,a_2) lens[(a_2)*4 + a_1 - 5]
#define ssize2_ref(a_1,a_2) ssize2[(a_2)*14 + a_1 - 15]
#define dt7_ref(a_1,a_2) dt7[(a_2)*4 + a_1 - 5]
#define dt8_ref(a_1,a_2,a_3) dt8[((a_3)*4 + (a_2))*7 + a_1 - 36]


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
		sx[i__ - 1] = dx1[i__ - 1];
		sy[i__ - 1] = dy1[i__ - 1];
/* L20: */
	    }

	    if (combla_1.icase == 1) {
		d__1 = ddot_(&combla_1.n, sx, &combla_1.incx, sy, &
			combla_1.incy);
		stest1_(&d__1, &dt7_ref(kn, ki), &ssize1[kn - 1], sfac);
	    } else if (combla_1.icase == 2) {
		daxpy_(&combla_1.n, &sa, sx, &combla_1.incx, sy, &
			combla_1.incy);
		i__1 = leny;
		for (j = 1; j <= i__1; ++j) {
		    sty[j - 1] = dt8_ref(j, kn, ki);
/* L40: */
		}
		stest_(&leny, sy, sty, &ssize2_ref(1, ksize), sfac);
	    } else if (combla_1.icase == 5) {
		for (i__ = 1; i__ <= 7; ++i__) {
		    sty[i__ - 1] = dt10y_ref(i__, kn, ki);
/* L60: */
		}
		dcopy_(&combla_1.n, sx, &combla_1.incx, sy, &combla_1.incy);
		stest_(&leny, sy, sty, &ssize2_ref(1, 1), &c_b34);
	    } else if (combla_1.icase == 6) {
		dswap_(&combla_1.n, sx, &combla_1.incx, sy, &combla_1.incy);
		for (i__ = 1; i__ <= 7; ++i__) {
		    stx[i__ - 1] = dt10x_ref(i__, kn, ki);
		    sty[i__ - 1] = dt10y_ref(i__, kn, ki);
/* L80: */
		}
		stest_(&lenx, sx, stx, &ssize2_ref(1, 1), &c_b34);
		stest_(&leny, sy, sty, &ssize2_ref(1, 1), &c_b34);
	    } else {
		s_wsle(&io___63);
		do_lio(&c__9, &c__1, " Shouldn't be here in CHECK2", (ftnlen)
			28);
		e_wsle();
		s_stop("", (ftnlen)0);
	    }
/* L100: */
	}
/* L120: */
    }
    return 0;
} /* check2_ */

#undef dt8_ref
#undef dt7_ref
#undef ssize2_ref
#undef lens_ref
#undef dt10y_ref
#undef dt10x_ref


/* Subroutine */ int check3_(doublereal *sfac)
{
    /* Initialized data */

    static integer incxs[4] = { 1,2,-2,-1 };
    static integer incys[4] = { 1,-2,1,-2 };
    static integer lens[8]	/* was [4][2] */ = { 1,1,2,4,1,1,3,7 };
    static integer ns[4] = { 0,1,2,4 };
    static doublereal dx1[7] = { .6,.1,-.5,.8,.9,-.3,-.4 };
    static doublereal dy1[7] = { .5,-.9,.3,.7,-.6,.2,.8 };
    static doublereal sc = .8;
    static doublereal ss = .6;
    static doublereal dt9x[112]	/* was [7][4][4] */ = { .6,0.,0.,0.,0.,0.,0.,
	    .78,0.,0.,0.,0.,0.,0.,.78,-.46,0.,0.,0.,0.,0.,.78,-.46,-.22,1.06,
	    0.,0.,0.,.6,0.,0.,0.,0.,0.,0.,.78,0.,0.,0.,0.,0.,0.,.66,.1,-.1,0.,
	    0.,0.,0.,.96,.1,-.76,.8,.9,-.3,-.02,.6,0.,0.,0.,0.,0.,0.,.78,0.,
	    0.,0.,0.,0.,0.,-.06,.1,-.1,0.,0.,0.,0.,.9,.1,-.22,.8,.18,-.3,-.02,
	    .6,0.,0.,0.,0.,0.,0.,.78,0.,0.,0.,0.,0.,0.,.78,.26,0.,0.,0.,0.,0.,
	    .78,.26,-.76,1.12,0.,0.,0. };
    static doublereal dt9y[112]	/* was [7][4][4] */ = { .5,0.,0.,0.,0.,0.,0.,
	    .04,0.,0.,0.,0.,0.,0.,.04,-.78,0.,0.,0.,0.,0.,.04,-.78,.54,.08,0.,
	    0.,0.,.5,0.,0.,0.,0.,0.,0.,.04,0.,0.,0.,0.,0.,0.,.7,-.9,-.12,0.,
	    0.,0.,0.,.64,-.9,-.3,.7,-.18,.2,.28,.5,0.,0.,0.,0.,0.,0.,.04,0.,
	    0.,0.,0.,0.,0.,.7,-1.08,0.,0.,0.,0.,0.,.64,-1.26,.54,.2,0.,0.,0.,
	    .5,0.,0.,0.,0.,0.,0.,.04,0.,0.,0.,0.,0.,0.,.04,-.9,.18,0.,0.,0.,
	    0.,.04,-.9,.18,.7,-.18,.2,.16 };
    static doublereal ssize2[28]	/* was [14][2] */ = { 0.,0.,0.,0.,0.,
	    0.,0.,0.,0.,0.,0.,0.,0.,0.,1.17,1.17,1.17,1.17,1.17,1.17,1.17,
	    1.17,1.17,1.17,1.17,1.17,1.17,1.17 };

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    static integer lenx, leny;
    static doublereal mwpc[11];
    extern /* Subroutine */ int drot_(integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *);
    static integer mwpn[11];
    static doublereal mwps[11], mwpx[5], mwpy[5];
    static integer i__, k, ksize;
    static doublereal copyx[5], copyy[5];
    extern /* Subroutine */ int stest_(integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *);
    static doublereal mwptx[55]	/* was [11][5] */, mwpty[55]	/* was [11][5]
	     */;
    static integer ki, kn, mx, my;
    static doublereal sx[7], sy[7];
    static integer mwpinx[11], mwpiny[11];
    static doublereal mwpstx[5], mwpsty[5], stx[7], sty[7];

    /* Fortran I/O blocks */
    static cilist io___88 = { 0, 6, 0, 0, 0 };



#define lens_ref(a_1,a_2) lens[(a_2)*4 + a_1 - 5]
#define mwptx_ref(a_1,a_2) mwptx[(a_2)*11 + a_1 - 12]
#define mwpty_ref(a_1,a_2) mwpty[(a_2)*11 + a_1 - 12]
#define ssize2_ref(a_1,a_2) ssize2[(a_2)*14 + a_1 - 15]
#define dt9x_ref(a_1,a_2,a_3) dt9x[((a_3)*4 + (a_2))*7 + a_1 - 36]
#define dt9y_ref(a_1,a_2,a_3) dt9y[((a_3)*4 + (a_2))*7 + a_1 - 36]


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

	    if (combla_1.icase == 4) {
		for (i__ = 1; i__ <= 7; ++i__) {
		    sx[i__ - 1] = dx1[i__ - 1];
		    sy[i__ - 1] = dy1[i__ - 1];
		    stx[i__ - 1] = dt9x_ref(i__, kn, ki);
		    sty[i__ - 1] = dt9y_ref(i__, kn, ki);
/* L20: */
		}
		drot_(&combla_1.n, sx, &combla_1.incx, sy, &combla_1.incy, &
			sc, &ss);
		stest_(&lenx, sx, stx, &ssize2_ref(1, ksize), sfac);
		stest_(&leny, sy, sty, &ssize2_ref(1, ksize), sfac);
	    } else {
		s_wsle(&io___88);
		do_lio(&c__9, &c__1, " Shouldn't be here in CHECK3", (ftnlen)
			28);
		e_wsle();
		s_stop("", (ftnlen)0);
	    }
/* L40: */
	}
/* L60: */
    }

    mwpc[0] = 1.;
    for (i__ = 2; i__ <= 11; ++i__) {
	mwpc[i__ - 1] = 0.;
/* L80: */
    }
    mwps[0] = 0.;
    for (i__ = 2; i__ <= 6; ++i__) {
	mwps[i__ - 1] = 1.;
/* L100: */
    }
    for (i__ = 7; i__ <= 11; ++i__) {
	mwps[i__ - 1] = -1.;
/* L120: */
    }
    mwpinx[0] = 1;
    mwpinx[1] = 1;
    mwpinx[2] = 1;
    mwpinx[3] = -1;
    mwpinx[4] = 1;
    mwpinx[5] = -1;
    mwpinx[6] = 1;
    mwpinx[7] = 1;
    mwpinx[8] = -1;
    mwpinx[9] = 1;
    mwpinx[10] = -1;
    mwpiny[0] = 1;
    mwpiny[1] = 1;
    mwpiny[2] = -1;
    mwpiny[3] = -1;
    mwpiny[4] = 2;
    mwpiny[5] = 1;
    mwpiny[6] = 1;
    mwpiny[7] = -1;
    mwpiny[8] = -1;
    mwpiny[9] = 2;
    mwpiny[10] = 1;
    for (i__ = 1; i__ <= 11; ++i__) {
	mwpn[i__ - 1] = 5;
/* L140: */
    }
    mwpn[4] = 3;
    mwpn[9] = 3;
    for (i__ = 1; i__ <= 5; ++i__) {
	mwpx[i__ - 1] = (doublereal) i__;
	mwpy[i__ - 1] = (doublereal) i__;
	mwptx_ref(1, i__) = (doublereal) i__;
	mwpty_ref(1, i__) = (doublereal) i__;
	mwptx_ref(2, i__) = (doublereal) i__;
	mwpty_ref(2, i__) = (doublereal) (-i__);
	mwptx_ref(3, i__) = (doublereal) (6 - i__);
	mwpty_ref(3, i__) = (doublereal) (i__ - 6);
	mwptx_ref(4, i__) = (doublereal) i__;
	mwpty_ref(4, i__) = (doublereal) (-i__);
	mwptx_ref(6, i__) = (doublereal) (6 - i__);
	mwpty_ref(6, i__) = (doublereal) (i__ - 6);
	mwptx_ref(7, i__) = (doublereal) (-i__);
	mwpty_ref(7, i__) = (doublereal) i__;
	mwptx_ref(8, i__) = (doublereal) (i__ - 6);
	mwpty_ref(8, i__) = (doublereal) (6 - i__);
	mwptx_ref(9, i__) = (doublereal) (-i__);
	mwpty_ref(9, i__) = (doublereal) i__;
	mwptx_ref(11, i__) = (doublereal) (i__ - 6);
	mwpty_ref(11, i__) = (doublereal) (6 - i__);
/* L160: */
    }
    mwptx_ref(5, 1) = 1.;
    mwptx_ref(5, 2) = 3.;
    mwptx_ref(5, 3) = 5.;
    mwptx_ref(5, 4) = 4.;
    mwptx_ref(5, 5) = 5.;
    mwpty_ref(5, 1) = -1.;
    mwpty_ref(5, 2) = 2.;
    mwpty_ref(5, 3) = -2.;
    mwpty_ref(5, 4) = 4.;
    mwpty_ref(5, 5) = -3.;
    mwptx_ref(10, 1) = -1.;
    mwptx_ref(10, 2) = -3.;
    mwptx_ref(10, 3) = -5.;
    mwptx_ref(10, 4) = 4.;
    mwptx_ref(10, 5) = 5.;
    mwpty_ref(10, 1) = 1.;
    mwpty_ref(10, 2) = 2.;
    mwpty_ref(10, 3) = 2.;
    mwpty_ref(10, 4) = 4.;
    mwpty_ref(10, 5) = 3.;
    for (i__ = 1; i__ <= 11; ++i__) {
	combla_1.incx = mwpinx[i__ - 1];
	combla_1.incy = mwpiny[i__ - 1];
	for (k = 1; k <= 5; ++k) {
	    copyx[k - 1] = mwpx[k - 1];
	    copyy[k - 1] = mwpy[k - 1];
	    mwpstx[k - 1] = mwptx_ref(i__, k);
	    mwpsty[k - 1] = mwpty_ref(i__, k);
/* L180: */
	}
	drot_(&mwpn[i__ - 1], copyx, &combla_1.incx, copyy, &combla_1.incy, &
		mwpc[i__ - 1], &mwps[i__ - 1]);
	stest_(&c__5, copyx, mwpstx, mwpstx, sfac);
	stest_(&c__5, copyy, mwpsty, mwpsty, sfac);
/* L200: */
    }
    return 0;
} /* check3_ */

#undef dt9y_ref
#undef dt9x_ref
#undef ssize2_ref
#undef mwpty_ref
#undef mwptx_ref
#undef lens_ref


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
    static cilist io___105 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___106 = { 0, 6, 0, fmt_99998, 0 };
    static cilist io___107 = { 0, 6, 0, fmt_99997, 0 };


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
	s_wsfe(&io___105);
	e_wsfe();
	s_wsfe(&io___106);
	e_wsfe();
L20:
	s_wsfe(&io___107);
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
    static cilist io___110 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___111 = { 0, 6, 0, fmt_99998, 0 };
    static cilist io___113 = { 0, 6, 0, fmt_99997, 0 };


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
    s_wsfe(&io___110);
    e_wsfe();
    s_wsfe(&io___111);
    e_wsfe();
L20:
    id = *icomp - *itrue;
    s_wsfe(&io___113);
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

   Main program alias */ int dblat1_ () { MAIN__ (); return 0; }

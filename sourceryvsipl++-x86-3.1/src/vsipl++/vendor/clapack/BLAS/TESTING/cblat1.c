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
static real c_b43 = 1.f;

/* Main program */ MAIN__(void)
{
    /* Initialized data */

    static real sfac = 9.765625e-4f;

    /* Format strings */
    static char fmt_99999[] = "(\002 Complex BLAS Test Program Results\002,/"
	    "1x)";
    static char fmt_99998[] = "(\002                                    ----"
	    "- PASS -----\002)";

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    extern /* Subroutine */ int check1_(real *), check2_(real *);
    static integer ic;
    extern /* Subroutine */ int header_(void);

    /* Fortran I/O blocks */
    static cilist io___2 = { 0, 6, 0, fmt_99999, 0 };
    static cilist io___4 = { 0, 6, 0, fmt_99998, 0 };


/*     Test program for the COMPLEX    Level 1 BLAS.   
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

    static char l[6*10] = "CDOTC " "CDOTU " "CAXPY " "CCOPY " "CSWAP " "SCNR"
	    "M2" "SCASUM" "CSCAL " "CSSCAL" "ICAMAX";

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


/* Subroutine */ int check1_(real *sfac)
{
    /* Initialized data */

    static real strue2[5] = { 0.f,.5f,.6f,.7f,.7f };
    static real strue4[5] = { 0.f,.7f,1.f,1.3f,1.7f };
    static complex ctrue5[80]	/* was [8][5][2] */ = { {.1f,.1f},{1.f,2.f},{
	    1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{-.16f,
	    -.37f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f}
	    ,{3.f,4.f},{-.17f,-.19f},{.13f,-.39f},{5.f,6.f},{5.f,6.f},{5.f,
	    6.f},{5.f,6.f},{5.f,6.f},{5.f,6.f},{.11f,-.03f},{-.17f,.46f},{
	    -.17f,-.19f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{
	    .19f,-.17f},{.32f,.09f},{.23f,-.24f},{.18f,.01f},{2.f,3.f},{2.f,
	    3.f},{2.f,3.f},{2.f,3.f},{.1f,.1f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{
	    4.f,5.f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{-.16f,-.37f},{6.f,7.f},{
	    6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{-.17f,
	    -.19f},{8.f,9.f},{.13f,-.39f},{2.f,5.f},{2.f,5.f},{2.f,5.f},{2.f,
	    5.f},{2.f,5.f},{.11f,-.03f},{3.f,6.f},{-.17f,.46f},{4.f,7.f},{
	    -.17f,-.19f},{7.f,2.f},{7.f,2.f},{7.f,2.f},{.19f,-.17f},{5.f,8.f},
	    {.32f,.09f},{6.f,9.f},{.23f,-.24f},{8.f,3.f},{.18f,.01f},{9.f,4.f}
	     };
    static complex ctrue6[80]	/* was [8][5][2] */ = { {.1f,.1f},{1.f,2.f},{
	    1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{.09f,
	    -.12f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f}
	    ,{3.f,4.f},{.03f,-.09f},{.15f,-.03f},{5.f,6.f},{5.f,6.f},{5.f,6.f}
	    ,{5.f,6.f},{5.f,6.f},{5.f,6.f},{.03f,.03f},{-.18f,.03f},{.03f,
	    -.09f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{.09f,
	    .03f},{.03f,.12f},{.12f,.03f},{.03f,.06f},{2.f,3.f},{2.f,3.f},{
	    2.f,3.f},{2.f,3.f},{.1f,.1f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{4.f,
	    5.f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{.09f,-.12f},{6.f,7.f},{6.f,
	    7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{.03f,
	    -.09f},{8.f,9.f},{.15f,-.03f},{2.f,5.f},{2.f,5.f},{2.f,5.f},{2.f,
	    5.f},{2.f,5.f},{.03f,.03f},{3.f,6.f},{-.18f,.03f},{4.f,7.f},{.03f,
	    -.09f},{7.f,2.f},{7.f,2.f},{7.f,2.f},{.09f,.03f},{5.f,8.f},{.03f,
	    .12f},{6.f,9.f},{.12f,.03f},{8.f,3.f},{.03f,.06f},{9.f,4.f} };
    static integer itrue3[5] = { 0,1,2,2,2 };
    static real sa = .3f;
    static complex ca = {.4f,-.7f};
    static complex cv[80]	/* was [8][5][2] */ = { {.1f,.1f},{1.f,2.f},{
	    1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{1.f,2.f},{.3f,
	    -.4f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},{3.f,4.f},
	    {3.f,4.f},{.1f,-.3f},{.5f,-.1f},{5.f,6.f},{5.f,6.f},{5.f,6.f},{
	    5.f,6.f},{5.f,6.f},{5.f,6.f},{.1f,.1f},{-.6f,.1f},{.1f,-.3f},{7.f,
	    8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{7.f,8.f},{.3f,.1f},{.1f,.4f},{
	    .4f,.1f},{.1f,.2f},{2.f,3.f},{2.f,3.f},{2.f,3.f},{2.f,3.f},{.1f,
	    .1f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{4.f,5.f},{
	    4.f,5.f},{.3f,-.4f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,7.f},{6.f,
	    7.f},{6.f,7.f},{6.f,7.f},{.1f,-.3f},{8.f,9.f},{.5f,-.1f},{2.f,5.f}
	    ,{2.f,5.f},{2.f,5.f},{2.f,5.f},{2.f,5.f},{.1f,.1f},{3.f,6.f},{
	    -.6f,.1f},{4.f,7.f},{.1f,-.3f},{7.f,2.f},{7.f,2.f},{7.f,2.f},{.3f,
	    .1f},{5.f,8.f},{.1f,.4f},{6.f,9.f},{.4f,.1f},{8.f,3.f},{.1f,.2f},{
	    9.f,4.f} };

    /* System generated locals */
    integer i__1, i__2, i__3;
    real r__1;
    complex q__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    static integer i__;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *, 
	    integer *), ctest_(integer *, complex *, complex *, complex *, 
	    real *);
    static complex mwpcs[5], mwpct[5];
    extern doublereal scnrm2_(integer *, complex *, integer *);
    extern /* Subroutine */ int itest1_(integer *, integer *), stest1_(real *,
	     real *, real *, real *);
    static complex cx[8];
    extern integer icamax_(integer *, complex *, integer *);
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer 
	    *);
    extern doublereal scasum_(integer *, complex *, integer *);
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
		r__1 = scnrm2_(&combla_1.n, cx, &combla_1.incx);
		stest1_(&r__1, &strue2[np1 - 1], &strue2[np1 - 1], sfac);
	    } else if (combla_1.icase == 7) {
		r__1 = scasum_(&combla_1.n, cx, &combla_1.incx);
		stest1_(&r__1, &strue4[np1 - 1], &strue4[np1 - 1], sfac);
	    } else if (combla_1.icase == 8) {
		cscal_(&combla_1.n, &ca, cx, &combla_1.incx);
		ctest_(&len, cx, &ctrue5_ref(1, np1, combla_1.incx), &
			ctrue5_ref(1, np1, combla_1.incx), sfac);
	    } else if (combla_1.icase == 9) {
		csscal_(&combla_1.n, &sa, cx, &combla_1.incx);
		ctest_(&len, cx, &ctrue6_ref(1, np1, combla_1.incx), &
			ctrue6_ref(1, np1, combla_1.incx), sfac);
	    } else if (combla_1.icase == 10) {
		i__1 = icamax_(&combla_1.n, cx, &combla_1.incx);
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
/*        CSCAL   
          Add a test for alpha equal to zero. */
	ca.r = 0.f, ca.i = 0.f;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    mwpct[i__1].r = 0.f, mwpct[i__1].i = 0.f;
	    i__1 = i__ - 1;
	    mwpcs[i__1].r = 1.f, mwpcs[i__1].i = 1.f;
/* L80: */
	}
	cscal_(&c__5, &ca, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
    } else if (combla_1.icase == 9) {
/*        CSSCAL   
          Add a test for alpha equal to zero. */
	sa = 0.f;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    mwpct[i__1].r = 0.f, mwpct[i__1].i = 0.f;
	    i__1 = i__ - 1;
	    mwpcs[i__1].r = 1.f, mwpcs[i__1].i = 1.f;
/* L100: */
	}
	csscal_(&c__5, &sa, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
/*        Add a test for alpha equal to one. */
	sa = 1.f;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    mwpct[i__1].r = cx[i__2].r, mwpct[i__1].i = cx[i__2].i;
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    mwpcs[i__1].r = cx[i__2].r, mwpcs[i__1].i = cx[i__2].i;
/* L120: */
	}
	csscal_(&c__5, &sa, cx, &combla_1.incx);
	ctest_(&c__5, cx, mwpct, mwpcs, sfac);
/*        Add a test for alpha equal to minus one. */
	sa = -1.f;
	for (i__ = 1; i__ <= 5; ++i__) {
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    q__1.r = -cx[i__2].r, q__1.i = -cx[i__2].i;
	    mwpct[i__1].r = q__1.r, mwpct[i__1].i = q__1.i;
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    q__1.r = -cx[i__2].r, q__1.i = -cx[i__2].i;
	    mwpcs[i__1].r = q__1.r, mwpcs[i__1].i = q__1.i;
/* L140: */
	}
	csscal_(&c__5, &sa, cx, &combla_1.incx);
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


/* Subroutine */ int check2_(real *sfac)
{
    /* Initialized data */

    static complex ca = {.4f,-.7f};
    static integer incxs[4] = { 1,2,-2,-1 };
    static integer incys[4] = { 1,-2,1,-2 };
    static integer lens[8]	/* was [4][2] */ = { 1,1,2,4,1,1,3,7 };
    static integer ns[4] = { 0,1,2,4 };
    static complex cx1[7] = { {.7f,-.8f},{-.4f,-.7f},{-.1f,-.9f},{.2f,-.8f},{
	    -.9f,-.4f},{.1f,.4f},{-.6f,.6f} };
    static complex cy1[7] = { {.6f,-.6f},{-.9f,.5f},{.7f,-.6f},{.1f,-.5f},{
	    -.1f,-.2f},{-.5f,-.3f},{.8f,-.7f} };
    static complex ct8[112]	/* was [7][4][4] */ = { {.6f,-.6f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.32f,-1.41f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.32f,
	    -1.41f},{-1.55f,.5f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{.32f,-1.41f},{-1.55f,.5f},{.03f,-.89f},{-.38f,-.96f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{0.f,0.f},{.32f,-1.41f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{-.07f,-.89f},{-.9f,.5f},{
	    .42f,-1.41f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.78f,.06f},{
	    -.9f,.5f},{.06f,-.13f},{.1f,-.5f},{-.77f,-.49f},{-.5f,-.3f},{.52f,
	    -1.51f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{.32f,-1.41f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{-.07f,-.89f},{-1.18f,-.31f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.78f,.06f},{-1.54f,.97f},{
	    .03f,-.89f},{-.18f,-1.31f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,
	    -.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {.32f,-1.41f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{.32f,-1.41f},{-.9f,.5f},{.05f,-.6f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{.32f,-1.41f},{-.9f,.5f},{.05f,-.6f},{.1f,
	    -.5f},{-.77f,-.49f},{-.5f,-.3f},{.32f,-1.16f} };
    static complex ct7[16]	/* was [4][4] */ = { {0.f,0.f},{-.06f,-.9f},{
	    .65f,-.47f},{-.34f,-1.22f},{0.f,0.f},{-.06f,-.9f},{-.59f,-1.46f},{
	    -1.04f,-.04f},{0.f,0.f},{-.06f,-.9f},{-.83f,.59f},{.07f,-.37f},{
	    0.f,0.f},{-.06f,-.9f},{-.76f,-1.15f},{-1.33f,-1.82f} };
    static complex ct6[16]	/* was [4][4] */ = { {0.f,0.f},{.9f,.06f},{
	    .91f,-.77f},{1.8f,-.1f},{0.f,0.f},{.9f,.06f},{1.45f,.74f},{.2f,
	    .9f},{0.f,0.f},{.9f,.06f},{-.55f,.23f},{.83f,-.39f},{0.f,0.f},{
	    .9f,.06f},{1.04f,.79f},{1.95f,1.22f} };
    static complex ct10x[112]	/* was [7][4][4] */ = { {.7f,-.8f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,-.6f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,-.6f},
	    {-.9f,.5f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,
	    -.6f},{-.9f,.5f},{.7f,-.6f},{.1f,-.5f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{.7f,-.8f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{.7f,-.6f},{-.4f,-.7f},{.6f,-.6f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{.8f,-.7f},{-.4f,-.7f},{-.1f,-.2f},{.2f,
	    -.8f},{.7f,-.6f},{.1f,.4f},{.6f,-.6f},{.7f,-.8f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,-.6f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{-.9f,.5f},{
	    -.4f,-.7f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{
	    .1f,-.5f},{-.4f,-.7f},{.7f,-.6f},{.2f,-.8f},{-.9f,.5f},{.1f,.4f},{
	    .6f,-.6f},{.7f,-.8f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{.6f,-.6f},{.7f,-.6f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{.6f,-.6f},{.7f,-.6f},{-.1f,-.2f},{
	    .8f,-.7f},{0.f,0.f},{0.f,0.f},{0.f,0.f} };
    static complex ct10y[112]	/* was [7][4][4] */ = { {.6f,-.6f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.7f,-.8f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.7f,-.8f},
	    {-.4f,-.7f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{
	    .7f,-.8f},{-.4f,-.7f},{-.1f,-.9f},{.2f,-.8f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{.7f,-.8f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{-.1f,-.9f},{-.9f,.5f},{.7f,-.8f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{-.6f,.6f},{-.9f,.5f},{-.9f,-.4f},{
	    .1f,-.5f},{-.1f,-.9f},{-.5f,-.3f},{.7f,-.8f},{.6f,-.6f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{.7f,-.8f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{-.1f,-.9f}
	    ,{.7f,-.8f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{
	    -.6f,.6f},{-.9f,-.4f},{-.1f,-.9f},{.7f,-.8f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{.6f,-.6f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{.7f,-.8f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},
	    {0.f,0.f},{0.f,0.f},{.7f,-.8f},{-.9f,.5f},{-.4f,-.7f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{.7f,-.8f},{-.9f,.5f},{-.4f,-.7f},{
	    .1f,-.5f},{-.1f,-.9f},{-.5f,-.3f},{.2f,-.8f} };
    static complex csize1[4] = { {0.f,0.f},{.9f,.9f},{1.63f,1.73f},{2.9f,
	    2.78f} };
    static complex csize3[14] = { {0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{
	    0.f,0.f},{0.f,0.f},{0.f,0.f},{1.17f,1.17f},{1.17f,1.17f},{1.17f,
	    1.17f},{1.17f,1.17f},{1.17f,1.17f},{1.17f,1.17f},{1.17f,1.17f} };
    static complex csize2[14]	/* was [7][2] */ = { {0.f,0.f},{0.f,0.f},{0.f,
	    0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{0.f,0.f},{1.54f,1.54f},{1.54f,
	    1.54f},{1.54f,1.54f},{1.54f,1.54f},{1.54f,1.54f},{1.54f,1.54f},{
	    1.54f,1.54f} };

    /* System generated locals */
    integer i__1, i__2;
    complex q__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    /* Subroutine */ int s_stop(char *, ftnlen);

    /* Local variables */
    static complex cdot[1];
    static integer lenx, leny, i__;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *, 
	    complex *, integer *);
    extern /* Complex */ VOID cdotu_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern /* Subroutine */ int cswap_(integer *, complex *, integer *, 
	    complex *, integer *), ctest_(integer *, complex *, complex *, 
	    complex *, real *);
    static integer ksize;
    extern /* Subroutine */ int caxpy_(integer *, complex *, complex *, 
	    integer *, complex *, integer *);
    static integer ki, kn;
    static complex cx[7], cy[7];
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
		cdotc_(&q__1, &combla_1.n, cx, &combla_1.incx, cy, &
			combla_1.incy);
		cdot[0].r = q__1.r, cdot[0].i = q__1.i;
		ctest_(&c__1, cdot, &ct6_ref(kn, ki), &csize1[kn - 1], sfac);
	    } else if (combla_1.icase == 2) {
		cdotu_(&q__1, &combla_1.n, cx, &combla_1.incx, cy, &
			combla_1.incy);
		cdot[0].r = q__1.r, cdot[0].i = q__1.i;
		ctest_(&c__1, cdot, &ct7_ref(kn, ki), &csize1[kn - 1], sfac);
	    } else if (combla_1.icase == 3) {
		caxpy_(&combla_1.n, &ca, cx, &combla_1.incx, cy, &
			combla_1.incy);
		ctest_(&leny, cy, &ct8_ref(1, kn, ki), &csize2_ref(1, ksize), 
			sfac);
	    } else if (combla_1.icase == 4) {
		ccopy_(&combla_1.n, cx, &combla_1.incx, cy, &combla_1.incy);
		ctest_(&leny, cy, &ct10y_ref(1, kn, ki), csize3, &c_b43);
	    } else if (combla_1.icase == 5) {
		cswap_(&combla_1.n, cx, &combla_1.incx, cy, &combla_1.incy);
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


/* Subroutine */ int stest_(integer *len, real *scomp, real *strue, real *
	ssize, real *sfac)
{
    /* Format strings */
    static char fmt_99999[] = "(\002                                       F"
	    "AIL\002)";
    static char fmt_99998[] = "(/\002 CASE  N INCX INCY MODE  I             "
	    "               \002,\002 COMP(I)                             TRU"
	    "E(I)  DIFFERENCE\002,\002     SIZE(I)\002,/1x)";
    static char fmt_99997[] = "(1x,i4,i3,3i5,i3,2e36.8,2e12.4)";

    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3, r__4, r__5;

    /* Builtin functions */
    integer s_wsfe(cilist *), e_wsfe(void), do_fio(integer *, char *, ftnlen);

    /* Local variables */
    static integer i__;
    extern doublereal sdiff_(real *, real *);
    static real sd;

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
	r__4 = (r__1 = ssize[i__], dabs(r__1)) + (r__2 = *sfac * sd, dabs(
		r__2));
	r__5 = (r__3 = ssize[i__], dabs(r__3));
	if (sdiff_(&r__4, &r__5) == 0.f) {
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
	do_fio(&c__1, (char *)&scomp[i__], (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&strue[i__], (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&sd, (ftnlen)sizeof(real));
	do_fio(&c__1, (char *)&ssize[i__], (ftnlen)sizeof(real));
	e_wsfe();
L40:
	;
    }
    return 0;

} /* stest_   

   Subroutine */ int stest1_(real *scomp1, real *strue1, real *ssize, real *
	sfac)
{
    static real scomp[1], strue[1];
    extern /* Subroutine */ int stest_(integer *, real *, real *, real *, 
	    real *);

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

doublereal sdiff_(real *sa, real *sb)
{
    /* System generated locals */
    real ret_val;

/*     ********************************* SDIFF **************************   
       COMPUTES DIFFERENCE OF TWO NUMBERS.  C. L. LAWSON, JPL 1974 FEB 15 */

    ret_val = *sa - *sb;
    return ret_val;
} /* sdiff_   

   Subroutine */ int ctest_(integer *len, complex *ccomp, complex *ctrue, 
	complex *csize, real *sfac)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    static integer i__;
    static real scomp[20], ssize[20], strue[20];
    extern /* Subroutine */ int stest_(integer *, real *, real *, real *, 
	    real *);

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
	scomp[(i__ << 1) - 1] = r_imag(&ccomp[i__]);
	i__2 = i__;
	strue[(i__ << 1) - 2] = ctrue[i__2].r;
	strue[(i__ << 1) - 1] = r_imag(&ctrue[i__]);
	i__2 = i__;
	ssize[(i__ << 1) - 2] = csize[i__2].r;
	ssize[(i__ << 1) - 1] = r_imag(&csize[i__]);
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

   Main program alias */ int cblat1_ () { MAIN__ (); return 0; }

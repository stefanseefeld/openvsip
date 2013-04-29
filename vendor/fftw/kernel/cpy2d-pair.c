/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

/* out of place copy routines for pairs of isomorphic 2D arrays */
#include "ifftw.h"

void X(cpy2d_pair)(R *I0, R *I1, R *O0, R *O1,
		   INT n0, INT is0, INT os0,
		   INT n1, INT is1, INT os1)
{
     INT i0, i1;

     for (i1 = 0; i1 < n1; ++i1)
	  for (i0 = 0; i0 < n0; ++i0) {
	       R x0 = I0[i0 * is0 + i1 * is1];
	       R x1 = I1[i0 * is0 + i1 * is1];
	       O0[i0 * os0 + i1 * os1] = x0;
	       O1[i0 * os0 + i1 * os1] = x1;
	  }
}

/* like cpy2d_pair, but read input contiguously if possible */
void X(cpy2d_pair_ci)(R *I0, R *I1, R *O0, R *O1,
		      INT n0, INT is0, INT os0,
		      INT n1, INT is1, INT os1)
{
     if (IABS(is0) < IABS(is1))	/* inner loop is for n0 */
	  X(cpy2d_pair) (I0, I1, O0, O1, n0, is0, os0, n1, is1, os1);
     else
	  X(cpy2d_pair) (I0, I1, O0, O1, n1, is1, os1, n0, is0, os0);
}

/* like cpy2d_pair, but write output contiguously if possible */
void X(cpy2d_pair_co)(R *I0, R *I1, R *O0, R *O1,
		      INT n0, INT is0, INT os0,
		      INT n1, INT is1, INT os1)
{
     if (IABS(os0) < IABS(os1))	/* inner loop is for n0 */
	  X(cpy2d_pair) (I0, I1, O0, O1, n0, is0, os0, n1, is1, os1);
     else
	  X(cpy2d_pair) (I0, I1, O0, O1, n1, is1, os1, n0, is0, os0);
}

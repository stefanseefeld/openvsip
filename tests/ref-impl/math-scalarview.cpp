/***********************************************************************

  File:   math-scalarview.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   04/21/2004

  Contents: Very, very simple tests of element-wise functions with
    scalar and view operands.

Copyright 2005 Georgia Tech Research Corporation, all rights reserved.

A non-exclusive, non-royalty bearing license is hereby granted to all
Persons to copy, distribute and produce derivative works for any
purpose, provided that this copyright notice and following disclaimer
appear on All copies: THIS LICENSE INCLUDES NO WARRANTIES, EXPRESSED
OR IMPLIED, WHETHER ORAL OR WRITTEN, WITH RESPECT TO THE SOFTWARE OR
OTHER MATERIAL INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE, OR ARISING
FROM A COURSE OF PERFORMANCE OR DEALING, OR FROM USAGE OR TRADE, OR OF
NON-INFRINGEMENT OF ANY PATENTS OF THIRD PARTIES. THE INFORMATION IN
THIS DOCUMENT SHOULD NOT BE CONSTRUED AS A COMMITMENT OF DEVELOPMENT
BY ANY OF THE ABOVE PARTIES.

The US Government has a license under these copyrights, and this
Material may be reproduced by or for the US Government.
***********************************************************************/

/***********************************************************************
  Included Files
***********************************************************************/

#include "test.hpp"
#include <vsip/domain.hpp>
#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/matrix.hpp>
#include <vsip/vector.hpp>
#include <cstdlib>

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int argc, char** argv)
{
  using namespace vsip;

  vsipl		init(argc, argv);

  /* Test multiplication.  */
  cscalar_f const
		scalar_cscalarf (4.0, 3.4);
  Vector<cscalar_f>
		vector_cscalar (2, cscalar_f (1.0, 1.0));
  insist (equal ((scalar_cscalarf * vector_cscalar).get (0),
		 cscalar_f (0.6, 7.4)));

  /* Test addition.  */
  insist (equal ((scalar_cscalarf + vector_cscalar).get (0),
		 cscalar_f (5.0, 4.4)));
  insist (equal ((vector_cscalar + scalar_cscalarf).get (0),
		 cscalar_f (5.0, 4.4)));

  /* Test ma, i.e., multiply-and-add.  */
  insist (equal
	  (ma (vector_cscalar, scalar_cscalarf, vector_cscalar).get (0),
	   cscalar_f (1.6, 8.4)));
  scalar_f const
		scalar_scalarf (4.0);
  Vector<scalar_f>
		vector_scalar (2, scalar_f (10.0));
  insist (equal
	  (ma (vector_scalar, scalar_scalarf, vector_scalar).get (0),
	   scalar_f (50.0)));

  /* Test subtraction.  */
  insist (equal
	  (sub (scalar_scalarf, vector_scalar).get (1),
	   scalar_f (-6.0)));
  insist (equal
	  ((scalar_scalarf - vector_scalar).get (1),
	   scalar_f (-6.0)));

  insist (equal
	  (sub (vector_scalar, scalar_scalarf).get (1),
	   scalar_f (6.0)));
  insist (equal
	  ((vector_scalar - scalar_scalarf).get (1),
	   scalar_f (6.0)));

  /* Test multiplication.  */
  insist (equal
	  (mul (scalar_scalarf, vector_scalar).get (1),
	   scalar_f (40.0)));
  insist (equal
	  ((scalar_scalarf * vector_scalar).get (1),
	   scalar_f (40.0)));

  insist (equal
	  (mul (vector_scalar, scalar_scalarf).get (1),
	   scalar_f (40.0)));
  insist (equal
	  ((vector_scalar * scalar_scalarf).get (1),
	   scalar_f (40.0)));

  return EXIT_SUCCESS;
}

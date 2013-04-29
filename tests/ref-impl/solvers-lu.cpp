/***********************************************************************

  File:   solvers-lu.cpp
  Author: Jeffrey Oldham, CodeSourcery, LLC.
  Date:   07/16/2004

  Contents: Very simple test of the LU decomposition class.

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
#include <stdlib.h>
#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/matrix.hpp>
#include <vsip/solvers.hpp>
#include <vsip/support.hpp>

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int argc, char** argv)
{
  using namespace vsip;

  vsipl	v(argc, argv);

  /* Create a matrix to decompose.  */

  length_type const
		columns = 2;
  length_type const
		rows = 2;
			/* The number of columns and rows for a matrix.  */

  Matrix<scalar_f>
		matrix (rows, columns);
			/* The matrix to decompose.  */

  matrix.put (0, 0, 2);
  matrix.put (0, 1, -2);
  matrix.put (1, 0, -2);
  matrix.put (1, 1, 5);

  /* Test instantiating other LU decomposition objects.  */
  
  Matrix<cscalar_f>
		cmatrix (rows, columns);
			/* A matrix to decompose.  */

  lud<cscalar_f, by_value>
		lud_cobject (cmatrix.size (0));

  lud<cscalar_f, by_reference>
		lud_cobject2 (cmatrix.size (0));

  lud<scalar_f, by_reference>
		lud_object2 (matrix.size (1));

  /* Decompose the matrix.  */

  typedef ::vsip::lud<scalar_f, by_value>
		lud_value_type;
			/* The LUD scalar_f by-value type.  */

  lud_value_type	lud_object_one (columns);

  /* Test the accessors.  */

  insist (equal (lud_object_one.length (), matrix.size (1)));

  /* Test copying.  */

  lud_value_type	lud_object_two (lud_object_one);
  insist (equal (lud_object_one.length (),
		 lud_object_two.length ()));

  /* Invoke system solution functions.  */

  if (!lud_object_one.decompose (matrix))
    return EXIT_FAILURE;

  lud_object_one.solve<mat_ntrans> (matrix);

  if (!lud_object_two.decompose (matrix))
    return EXIT_FAILURE;

  lud_object_two.solve<mat_trans> (matrix);

  /* Test solving a particular system for scalar_f, by_value.  */
  Matrix<scalar_f>
		sA (6,6);
			/* The left-hand side which may be decomposed.  */
  Matrix<scalar_f>
		sA2 (6,6);
			/* The left-hand side which is not decomposed.  */
  Matrix<scalar_f>
		ssolution (6,3);
			/* The solution.  */
  Matrix<scalar_f>
		sB (6,3);
			/* The right-hand side.  */

  sA.put (0, 0, 0.5);
  sA.put (0, 1, 7.0);
  sA.put (0, 2, 10.0);
  sA.put (0, 3, 12.0);
  sA.put (0, 4, -3.0);
  sA.put (0, 5, 0.0);

  sA.put (1, 0, 2.0);
  sA.put (1, 1, 13.0);
  sA.put (1, 2, 18.0);
  sA.put (1, 3, 6.0);
  sA.put (1, 4, 0.0);
  sA.put (1, 5, 130.0);

  sA.put (2, 0, 3.0);
  sA.put (2, 1, -9.0);
  sA.put (2, 2, 2.0);
  sA.put (2, 3, 3.0);
  sA.put (2, 4, 2.0);
  sA.put (2, 5, -9.0);

  sA.put (3, 0, 4.0);
  sA.put (3, 1, 2.0);
  sA.put (3, 2, 2.0);
  sA.put (3, 3, 4.0);
  sA.put (3, 4, 1.0);
  sA.put (3, 5, 2.0);

  sA.put (4, 0, 0.2);
  sA.put (4, 1, 2.0);
  sA.put (4, 2, 9.0);
  sA.put (4, 3, 4.0);
  sA.put (4, 4, 1.0);
  sA.put (4, 5, 2.0);

  sA.put (5, 0, 0.1);
  sA.put (5, 1, 2.0);
  sA.put (5, 2, 0.3);
  sA.put (5, 3, 4.0);
  sA.put (5, 4, 1.0);
  sA.put (5, 5, 2.0);

  sA2 = sA;

  sB.put (0, 0, 77.85);
  sB.put (0, 1, 155.7);
  sB.put (0, 2, 311.4);

  sB.put (1, 0, 942.0);
  sB.put (1, 1, 1884.0);
  sB.put (1, 2, 3768.0);

  sB.put (2, 0, 1.0);
  sB.put (2, 1, 2.0);
  sB.put (2, 2, 4.0);

  sB.put (3, 0, 68.0);
  sB.put (3, 1, 136.0);
  sB.put (3, 2, 272.0);

  sB.put (4, 0, 85.2);
  sB.put (4, 1, 170.4);
  sB.put (4, 2, 340.8);

  sB.put (5, 0, 59.0);
  sB.put (5, 1, 118.0);
  sB.put (5, 2, 236.0);

  lud_value_type	lud_test (6);
  if (!lud_test.decompose (sA))
    return EXIT_FAILURE;
  Matrix<scalar_f>
		X (lud_test.solve<mat_ntrans> (sB));

  Matrix<scalar_f>
		difference (vsip::prod (sA2, X) - sB);
  for (vsip::index_type i = 0; i < 6; ++i)
    for (vsip::index_type j = 0; j < 3; ++j) {
      if (vsip::mag (difference.get (i,j)) > 0.01)
	return EXIT_FAILURE;
    }

  /* Test solving a particular system for cscalar_f, by_reference.  */

  Matrix<cscalar_f>
		A (7,7);
			/* The left-hand side which may be decomposed.  */
  Matrix<cscalar_f>
		A2 (7,7);
			/* The left-hand side which is not decomposed.  */
  Matrix<cscalar_f>
		solution (7,3);
  			/* The computed answer.  */
  Matrix<cscalar_f>
		B (7,3);
			/* The right-hand side.  */

  A.put (0, 0, cscalar_f (0.5, 0.1));
  A.put (0, 1, cscalar_f (7.0, 0.1));
  A.put (0, 2, cscalar_f (10.0, 0.1));
  A.put (0, 3, cscalar_f (12.0, 0.1));
  A.put (0, 4, cscalar_f (-3.0, 0.1));
  A.put (0, 5, cscalar_f (0.0, 0.1));
  A.put (0, 6, cscalar_f (0.05, 0.1));

  A.put (1, 0, cscalar_f (2.0, 0.1));
  A.put (1, 1, cscalar_f (13.0, 0.10));
  A.put (1, 2, cscalar_f (18.0, 0.10));
  A.put (1, 3, cscalar_f (6.0, 0.10));
  A.put (1, 4, cscalar_f (0.0, 0.1));
  A.put (1, 5, cscalar_f (130.0, 0.1));
  A.put (1, 6, cscalar_f (8.0, 0.1));

  A.put (2, 0, cscalar_f (3.0, 0.1));
  A.put (2, 1, cscalar_f (-9.0, 0.1));
  A.put (2, 2, cscalar_f (2.0, 0.1));
  A.put (2, 3, cscalar_f (3.0, 0.1));
  A.put (2, 4, cscalar_f (2.0, 0.2));
  A.put (2, 5, cscalar_f (-9.0, 0.2));
  A.put (2, 6, cscalar_f (6.0, 0.2));

  A.put (3, 0, cscalar_f (4.0, -0.2));
  A.put (3, 1, cscalar_f (2.0, 0.2));
  A.put (3, 2, cscalar_f (2.0, 0.2));
  A.put (3, 3, cscalar_f (4.0, 0.2));
  A.put (3, 4, cscalar_f (1.0, 0.2));
  A.put (3, 5, cscalar_f (2.0, 0.2));
  A.put (3, 6, cscalar_f (3.0, 0.2));

  A.put (4, 0, cscalar_f (0.2, 0.3));
  A.put (4, 1, cscalar_f (2.0, 0.3));
  A.put (4, 2, cscalar_f (9.0, 0.3));
  A.put (4, 3, cscalar_f (4.0, 0.3));
  A.put (4, 4, cscalar_f (1.0, 0.3));
  A.put (4, 5, cscalar_f (2.0, 0.3));
  A.put (4, 6, cscalar_f (3.0, 0.3));

  A.put (5, 0, cscalar_f (0.1, 0.4));
  A.put (5, 1, cscalar_f (2.0, 0.4));
  A.put (5, 2, cscalar_f (0.3, 0.4));
  A.put (5, 3, cscalar_f (4.0, 0.4));
  A.put (5, 4, cscalar_f (1.0, 0.4));
  A.put (5, 5, cscalar_f (2.0, 0.4));
  A.put (5, 6, cscalar_f (3.0, 0.4));

  A.put (6, 0, cscalar_f (0.0, 0.4));
  A.put (6, 1, cscalar_f (0.2, 0.4));
  A.put (6, 2, cscalar_f (3.0, 0.4));
  A.put (6, 3, cscalar_f (4.0, 0.4));
  A.put (6, 4, cscalar_f (1.0, 0.4));
  A.put (6, 5, cscalar_f (2.0, 0.4));
  A.put (6, 6, cscalar_f (3.0, 0.4));

  A2 = A;

  B.put (0, 0, cscalar_f (77.85, 4.5));
  B.put (0, 1, cscalar_f (155.70, 1.7));
  B.put (0, 2, cscalar_f (311.4, -3.4));

  B.put (1, 0, cscalar_f (942.0, 3.7));
  B.put (1, 1, cscalar_f (1884.0, 184.0));
  B.put (1, 2, cscalar_f (3768.0, -2.0));

  B.put (2, 0, cscalar_f (1.0, 1.0));
  B.put (2, 1, cscalar_f (2.0, 3.0));
  B.put (2, 2, cscalar_f (4.0, 2.0));

  B.put (3, 0, cscalar_f (68.0, 68.0));
  B.put (3, 1, cscalar_f (136.0, 16.0));
  B.put (3, 2, cscalar_f (272.0, 272.0));

  B.put (4, 0, cscalar_f (85.2, 85.2));
  B.put (4, 1, cscalar_f (170.4, 1170.4));
  B.put (4, 2, cscalar_f (340.8, 340.8));

  B.put (5, 0, cscalar_f (59.0, 59.0));
  B.put (5, 1, cscalar_f (118.0, 18.5));
  B.put (5, 2, cscalar_f (236.0, 62.0));

  B.put (6, 0, cscalar_f (5.0, 59.0));
  B.put (6, 1, cscalar_f (18.0, 11.6));
  B.put (6, 2, cscalar_f (6.0, 26.0));

  lud<cscalar_f, by_reference>
		lud_cobject_sys (A.size (0));

  lud_cobject_sys.decompose (A);
  lud_cobject_sys.solve<mat_ntrans> (B, solution);
  Matrix<cscalar_f>
		cdifference (vsip::prod (A2, solution) - B);
  /* Ensure the computed solution is close to the actual answer.  */
  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j) {
      if (std::abs (cdifference.get (i,j)) > 0.01)
	return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}

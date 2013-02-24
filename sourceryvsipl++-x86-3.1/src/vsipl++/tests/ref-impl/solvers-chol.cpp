/***********************************************************************

  File:   solvers-chol.cpp
  Author: Jeffrey Oldham, CodeSourcery, LLC.
  Date:   06/22/2004

  Contents: Very simple test of the Cholesky decomposition class.

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

  /* Test instantiating other Cholesky decomposition objects.  */
  
  Matrix<cscalar_f>
		cmatrix (rows, columns);
			/* A matrix to decompose.  */

  chold<cscalar_f, by_value>
		chold_cobject (upper, cmatrix.size (0));

  chold<cscalar_f, by_reference>
		chold_cobject2 (lower, cmatrix.size (0));

  chold<scalar_f, by_reference>
		chold_object2 (upper, matrix.size (1));

  /* Decompose the matrix.  */

  typedef ::vsip::chold<scalar_f, by_value>
		chold_value_type;
			/* The CHOLD scalar_f by-value type.  */

  chold_value_type	chold_object_one (lower, columns);

  /* Test the accessors.  */

  insist (equal (chold_object_one.uplo (), lower));
  insist (equal (chold_object_one.length (), matrix.size (1)));

  /* Test copying.  */

  chold_value_type	chold_object_two (chold_object_one);
  insist (equal (chold_object_one.uplo (), chold_object_two.uplo ()));
  insist (equal (chold_object_one.length (),
		 chold_object_two.length ()));

  /* Invoke system solution functions.  */

  if (!chold_object_one.decompose (matrix))
    return EXIT_FAILURE;

  chold_object_one.solve (matrix);

  if (!chold_object_two.decompose (matrix))
    return EXIT_FAILURE;

  chold_object_two.solve (matrix);

  /* Test solving a particular system for cscalar_f, by_reference.  */

  Matrix<cscalar_f>
		A (4,4);	/* The left-hand side.  */
  Matrix<cscalar_f>
		ans (4,3);	/* The (approximate) correct answer.  */
  Matrix<cscalar_f>
		solution (4,3);	/* The computed answer.  */
  Matrix<cscalar_f>
		B (4,3);	/* The right-hand side.  */

  A.put (0, 0, cscalar_f (1.0, 0.0));
  A.put (0, 1, cscalar_f (-2.0, 2.0));
  A.put (0, 2, cscalar_f (3.0, 2.0));
  A.put (0, 3, cscalar_f (1.0, 1.0));

  A.put (1, 0, cscalar_f (-2.0, -2.0));
  A.put (1, 1, cscalar_f (12.0, 0.0));
  A.put (1, 2, cscalar_f (6.0, -6.0));
  A.put (1, 3, cscalar_f (-2.0, -12.0));

  A.put (2, 0, cscalar_f (3.0, -2.0));
  A.put (2, 1, cscalar_f (6.0, 6.0));
  A.put (2, 2, cscalar_f (49.0, 0.0));
  A.put (2, 3, cscalar_f (5.0, -5.0));

  A.put (3, 0, cscalar_f (1.0, -1.0));
  A.put (3, 1, cscalar_f (-2.0, 12.0));
  A.put (3, 2, cscalar_f (5.0, 5.0));
  A.put (3, 3, cscalar_f (68.0, 0.0));

  B.put (0, 0, cscalar_f (1.0, 1.0));
  B.put (0, 1, cscalar_f (2.0, 0.50));
  B.put (0, 2, cscalar_f (3.0, 3.2));

  B.put (1, 0, cscalar_f (0.0, 2.0));
  B.put (1, 1, cscalar_f (1.0, 0.0));
  B.put (1, 2, cscalar_f (2.0, 0.6));

  B.put (2, 0, cscalar_f (3.0, 0.6));
  B.put (2, 1, cscalar_f (0.0, 2.0));
  B.put (2, 2, cscalar_f (1.0, 0.0));

  B.put (3, 0, cscalar_f (3.0, 5.0));
  B.put (3, 1, cscalar_f (4.0, 7.0));
  B.put (3, 2, cscalar_f (5.0, 8.0));

  ans.put (0, 0, cscalar_f (13.62, 19.50));
  ans.put (0, 1, cscalar_f (27.55, 5.37));
  ans.put (0, 2, cscalar_f (43.16, 40.96));

  ans.put (1, 0, cscalar_f (-0.11, 6.73));
  ans.put (1, 1, cscalar_f (5.24, 5.49));
  ans.put (1, 2, cscalar_f (3.33, 15.88));

  ans.put (2, 0, cscalar_f (-0.84, -1.40));
  ans.put (2, 1, cscalar_f (-1.94, -0.38));
  ans.put (2, 2, cscalar_f (-2.98, -2.97));

  ans.put (3, 0, cscalar_f (0.70, 0.37));
  ans.put (3, 1, cscalar_f (0.81, -0.16));
  ans.put (3, 2, cscalar_f (1.74, 0.47));

  chold<cscalar_f, by_reference>
		chold_cobject_sys (lower, A.size (0));

  chold_cobject_sys.decompose (A);
  chold_cobject_sys.solve (B, solution);

  /* Ensure the computed solution is close to the actual answer.  */
  for (vsip::index_type i = 0; i < 4; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      if (std::abs (ans.get (i,j) - solution.get (i,j)) > 0.01)
	return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

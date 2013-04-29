/***********************************************************************

  File:   solvers-qr.cpp
  Author: Jeffrey Oldham, CodeSourcery, LLC.
  Date:   09/19/2003

  Contents: Very simple test of the QR decomposition class.

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
		columns = 33;
  length_type const
		rows = 34;
			/* The number of columns and rows for a matrix.  */

  Matrix<scalar_f>
		matrix (rows, columns);
			/* The matrix to decompose.  */

  for (vsip::index_type row = matrix.size (0); row-- > 0; )
    for (vsip::index_type column = matrix.size (1); column-- > 0; )
      matrix.put (row, column, row * 7 + column);

  /* Decompose the matrix.  */

  typedef ::vsip::qrd<scalar_f, by_value>
		qrd_value_type;
			/* The QRD scalar_f by-value type.  */

  qrd_value_type	qrd_object_one (matrix.size (0), matrix.size (1),
				qrd_saveq);

  /* Test the accessors.  */

  insist (equal (qrd_object_one.rows (), matrix.size (0)));
  insist (equal (qrd_object_one.columns (), matrix.size (1)));
  insist (equal (qrd_object_one.qstorage (), qrd_saveq));

  /* Test copying.  */

  qrd_value_type	qrd_object_two (qrd_object_one);
  insist (equal (qrd_object_one.rows (), qrd_object_two.rows ()));
  insist (equal (qrd_object_one.columns (), qrd_object_two.columns ()));
  insist (equal (qrd_object_one.qstorage (),
		 qrd_object_two.qstorage ()));

  /* Invoke system solution functions.  */

  length_type const
		operand_length = 23;
			/* Size of the operand to multiply with Q.  */

  Matrix<scalar_f>
		empty_matrix (qrd_object_one.rows (), operand_length, 0.0);
			/* A Matrix to multiply with Q.  */

  insist (qrd_object_one.decompose (matrix));
  qrd_object_one.prodq<mat_ntrans, mat_lside> (empty_matrix);

  Matrix<scalar_f>
		empty_matrix2 (operand_length, qrd_object_one.rows (), 0.0);
			/* A Matrix to multiply with Q.  */

  Matrix<>	answer_prodq
    (qrd_object_one.
     prodq<mat_trans, mat_rside> (empty_matrix2));
			/* The product of Q and empty_matrix2.  */

  Matrix<scalar_f>
		empty_matrix3 (qrd_object_one.columns (), operand_length, 0.0);
			/* A Matrix to multiply with R.  */

  Matrix<>	answer_rsol
    (qrd_object_one.rsol<mat_trans> (empty_matrix3, 1.0));

  /* Test instantiating other QR decomposition objects.  */
  
  Matrix<cscalar_f>
		cmatrix (rows, columns);
			/* A matrix to decompose.  */

  qrd<cscalar_f, by_value>
		qrd_cobject (cmatrix.size (0), cmatrix.size (1),
			     qrd_saveq);

  qrd<cscalar_f, by_reference>
		qrd_cobject2 (cmatrix.size (0), cmatrix.size (1),
			     qrd_saveq);

  qrd<scalar_f, by_reference>
		qrd_object2 (matrix.size (0), matrix.size (1),
			     qrd_saveq);

  return EXIT_SUCCESS;
}

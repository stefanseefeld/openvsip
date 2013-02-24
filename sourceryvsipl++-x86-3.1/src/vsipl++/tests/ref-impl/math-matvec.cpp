/***********************************************************************

  File:   math-matvec.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   07/14/2003

  Contents: Very, very simple tests of matrix and vector math
    functions.

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
#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/matrix.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <cstdlib>

/***********************************************************************
  Inline Function Definitions
***********************************************************************/

inline
vsip::scalar_f
matrix_entry (vsip::index_type const i,
	      vsip::index_type const j)
{
  return i * 1000 + j;
}
			/* Returns a scalar_f matrix value to store at
			   (i,j).  */

inline
vsip::cscalar_f
cmatrix_entry (vsip::index_type const i,
	      vsip::index_type const j)
{
  return vsip::cscalar_f (i * 1000 + j, i * 1000 + j);
}
			/* Returns a complex matrix value to store at
			   (i,j).  */

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);

  vsip::Vector<vsip::cscalar_f>
		vector_cscalarf (2, vsip::cscalar_f (1.0, 17.0));
  vsip::Vector<vsip::scalar_f>
		vector_scalarf (2, static_cast<vsip::scalar_f>(3.0));

  /* Test dot products.  */

  // Test cvjdot.
  insist (equal (vsip::cvjdot (vector_cscalarf, vector_cscalarf),
		 vsip::cscalar_f (580.0, 0.0)));

  // Test dot.
  insist (equal (vsip::dot (vector_scalarf, vector_scalarf),
		 static_cast<vsip::scalar_f>(18.0)));
  insist (equal (vsip::dot (vector_cscalarf, vector_cscalarf),
		 vsip::cscalar_f (-576.0, 68.0)));

  /* Test transpositions.  */

  // Test trans.

  vsip::length_type const
		number_rows = 13;
			/* The number of rows in the original
			   matrix.  */

  vsip::length_type const
		number_columns = 8;
			/* The number of columns in the original
			   matrix.  */

  vsip::Matrix<>
		trans_input (number_rows, number_columns);

  for (vsip::index_type i = number_rows; i-- > 0; )
    for (vsip::index_type j = number_columns; j-- > 0; )
      trans_input.put (i, j, matrix_entry (i, j));

  vsip::Matrix<>
		trans_answer (vsip::trans (trans_input));

  for (vsip::index_type i = number_rows; i-- > 0; )
    for (vsip::index_type j = number_columns; j-- > 0; )
      insist (equal (trans_answer.get (j, i), matrix_entry (i, j)));

  // Test herm.

  vsip::Matrix<vsip::cscalar_f>
		herm_input (number_rows, number_columns);

  for (vsip::index_type i = number_rows; i-- > 0; )
    for (vsip::index_type j = number_columns; j-- > 0; )
      herm_input.put (i, j, cmatrix_entry (i, j));

  vsip::Matrix<vsip::cscalar_f>
		herm_answer (vsip::herm (herm_input));

  for (vsip::index_type i = number_rows; i-- > 0; )
    for (vsip::index_type j = number_columns; j-- > 0; )
      insist (equal (herm_answer.get (j, i),
		     vsip::conj (cmatrix_entry (i, j))));

  /* Test Kronecker tensor product.  */

  vector_scalarf = 1.0;
  vector_scalarf.put (0, 2.0);
  vsip::Matrix<>
		kron_answer (vsip::kron (static_cast<vsip::scalar_f>(7.0),
					 vector_scalarf, vector_scalarf));
  insist (equal (kron_answer.get (0,0),
		 static_cast<vsip::scalar_f>(7.0 * vector_scalarf.get (0) *
					     vector_scalarf.get (0))));
  insist (equal (kron_answer.get (1,0),
		 static_cast<vsip::scalar_f>(7.0 * vector_scalarf.get (1) *
					     vector_scalarf.get (0))));

  /* Test outer product.  */
  vector_scalarf = 3.0;
  insist (equal (vsip::outer (static_cast<vsip::scalar_f>(1.0),
			      vector_scalarf, vector_scalarf).get (0,0),
		 static_cast<vsip::scalar_f>(9.0)));
  insist (equal (vsip::outer (static_cast<vsip::scalar_f>(2.0),
			      vector_scalarf, vector_scalarf).get (0,0),
		 static_cast<vsip::scalar_f>(18.0)));
  insist (equal (vsip::outer (vsip::cscalar_f (2.0, 0.0),
			      vector_cscalarf, vector_cscalarf).get (0,0),
		 vsip::cscalar_f (580.0, 0.0)));

  /* Test matrix and vector products.  */

  vsip::Matrix<vsip::scalar_f>
		matrix_scalarf_one (3, 3, static_cast<vsip::scalar_f>(3.0));
  vsip::Matrix<vsip::scalar_f>
		matrix_scalarf_two (3, 3, static_cast<vsip::scalar_f>(-1.0));
  vsip::Matrix<vsip::scalar_f>
		matrix_scalarf_three (3, 3, static_cast<vsip::scalar_f>(0.0));
  vsip::Vector<vsip::scalar_f>
		vector_scalarf_one (3, static_cast<vsip::scalar_f>(-1.0));
  vsip::Vector<vsip::scalar_f>
		vector_scalarf_two (3, static_cast<vsip::scalar_f>(0.0));

  vsip::Matrix<vsip::cscalar_f>
		matrix_cscalarf_one (3, 3, vsip::cscalar_f(3.0, 0.0));
  vsip::Matrix<vsip::cscalar_f>
		matrix_cscalarf_two (3, 3, vsip::cscalar_f(-1.0, 0.0));
  vsip::Matrix<vsip::cscalar_f>
		matrix_cscalarf_three (3, 3, vsip::cscalar_f(0.0, 0.0));
  vsip::Vector<vsip::cscalar_f>
		vector_cscalarf_one (3, vsip::cscalar_f(-1.0, 0.0));
  vsip::Vector<vsip::cscalar_f>
		vector_cscalarf_two (3, vsip::cscalar_f(0.0, 0.0));

  // Test prod of a matrix and a matrix.

  matrix_scalarf_three = vsip::prod (matrix_scalarf_one,
				     matrix_scalarf_two);
  check_entry (matrix_scalarf_three, 2, 2,
	       static_cast<vsip::scalar_f>(-9.0));
  matrix_cscalarf_three = vsip::prod (matrix_cscalarf_one,
				      matrix_cscalarf_two);
  check_entry (matrix_cscalarf_three, 2, 2,
	       vsip::cscalar_f(-9.0, 0.0));

  // Test prod of a matrix and a vector.

  vector_scalarf = -1.0;
  vector_scalarf_two = vsip::prod (matrix_scalarf_one,
				   vector_scalarf_one);
  check_entry (vector_scalarf_two, 2,
	       static_cast<vsip::scalar_f>(-9.0));
  vector_cscalarf = vsip::cscalar_f (-1.0, 0.0);
  vector_cscalarf_two = vsip::prod (matrix_cscalarf_one,
				    vector_cscalarf_one);
  check_entry (vector_cscalarf_two, 2,
	       vsip::cscalar_f(-9.0, 0.0));

  // Test prod of a vector and a matrix.

  vector_scalarf = -1.0;
  vector_scalarf_two = vsip::prod (vector_scalarf_one,
				   matrix_scalarf_one);
  check_entry (vector_scalarf_two, 2,
	       static_cast<vsip::scalar_f>(-9.0));
  vector_cscalarf = vsip::cscalar_f (-1.0, 0.0);
  vector_cscalarf_two = vsip::prod (vector_cscalarf_one,
				    matrix_cscalarf_one);
  check_entry (vector_cscalarf_two, 2,
	       vsip::cscalar_f(-9.0, 0.0));

  // Test prod3 of a matrix and a matrix.

  matrix_scalarf_three = vsip::prod3 (matrix_scalarf_one,
				      matrix_scalarf_two);
  check_entry (matrix_scalarf_three, 2, 2,
	       static_cast<vsip::scalar_f>(-9.0));

  // Test prod3 of a matrix and a vector.

  vector_cscalarf = vsip::cscalar_f (-1.0, 0.0);
  vector_cscalarf_two = vsip::prod3 (matrix_cscalarf_one,
				     vector_cscalarf_one);
  check_entry (vector_cscalarf_two, 2,
	       vsip::cscalar_f(-9.0, 0.0));

  // Test prodh of a matrix and a matrix.

  matrix_cscalarf_one = vsip::cscalar_f (3.0, 1.0);
  matrix_cscalarf_two = vsip::cscalar_f (-1.0, -1.0);
  matrix_cscalarf_three = vsip::prodh (matrix_cscalarf_one,
				       matrix_cscalarf_two);
  check_entry (matrix_cscalarf_three, 2, 2,
	       vsip::cscalar_f(-12.0, 6.0));

  /* Test generalized matrix products and sums.  */

  using namespace vsip;
  cscalar_f const
		alpha (1.5, 0.25);
  Matrix<cscalar_f>
		a (3, 5);
  Matrix<cscalar_f>
		b (3, 5);
  cscalar_f const
		beta (-0.5, 2.0);
  Matrix<cscalar_f>
		c (3, 3);

  a.put (0, 0, cscalar_f (1.0, 0.1));
  a.put (0, 1, cscalar_f (2.0, 2.1));
  a.put (0, 2, cscalar_f (-3.0, -2.0));
  a.put (0, 3, cscalar_f (4.0, 3.0));
  a.put (0, 4, cscalar_f (5.0, 5.0));

  a.put (1, 0, cscalar_f (5.0, 3.0));
  a.put (1, 1, cscalar_f (0.1, 1.1));
  a.put (1, 2, cscalar_f (0.2, 1.2));
  a.put (1, 3, cscalar_f (0.3, -5.3));
  a.put (1, 4, cscalar_f (0.4, 1.4));

  a.put (2, 0, cscalar_f (-4.0, -2.0));
  a.put (2, 1, cscalar_f (3.0, 2.2));
  a.put (2, 2, cscalar_f (2.0, 2.2));
  a.put (2, 3, cscalar_f (0.0, 0.5));
  a.put (2, 4, cscalar_f (-1.0, 1.1));

  b.put (0, 0, cscalar_f (0.4, 1.4));
  b.put (0, 1, cscalar_f (1.5, 1.2));
  b.put (0, 2, cscalar_f (-2.7, -1.7));
  b.put (0, 3, cscalar_f (3.0, 3.0));
  b.put (0, 4, cscalar_f (9.0, 9.0));

  b.put (1, 0, cscalar_f (-1.1, -1.1));
  b.put (1, 1, cscalar_f (-0.2, -3.1));
  b.put (1, 2, cscalar_f (-0.3, -1.3));
  b.put (1, 3, cscalar_f (-0.2, -0.2));
  b.put (1, 4, cscalar_f (1.3, 1.3));

  b.put (2, 0, cscalar_f (3.0, 2.2));
  b.put (2, 1, cscalar_f (2.0, 2.1));
  b.put (2, 2, cscalar_f (1.0, 1.1));
  b.put (2, 3, cscalar_f (4.0, 40.0));
  b.put (2, 4, cscalar_f (-1.0, -1.0));

  for (vsip::index_type i = 3; i-- > 0; )
    for (vsip::index_type j = 3; j-- > 0; )
      c.put (i, j, cscalar_f (1.0, 0.5));

  gemp<mat_ntrans, mat_herm>(alpha, a, b, beta, c);
  
  /* Test elementwise vector-matrix products.  */

  vector_scalarf_one = -2.0;
  matrix_scalarf_one = 2.1;
  vector_cscalarf = vsip::cscalar_f (-1.0, 0.0);
  matrix_cscalarf_one = vsip::cscalar_f (3.0, 1.0);
  matrix_scalarf_two = vsip::vmmul<0> (vector_scalarf_one,
				       matrix_scalarf_one);
  check_entry (matrix_scalarf_two, 2, 2, 
	       static_cast<vsip::scalar_f>(-4.2));
  check_entry (vsip::vmmul<0> (vector_scalarf_one,
			       matrix_cscalarf_one),
	       2, 2, vsip::cscalar_f(-6.0, -2.0));
  check_entry (vsip::vmmul<1> (vector_cscalarf_one,
			       matrix_cscalarf_one),
	       2, 2, vsip::cscalar_f(-3.0, -1.0));

  return EXIT_SUCCESS;
}

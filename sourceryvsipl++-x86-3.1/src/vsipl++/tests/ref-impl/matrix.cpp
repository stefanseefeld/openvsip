/***********************************************************************

  File:   matrix.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   08/12/2002

  Contents: Very, very simple tests of the Matrix view class.

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

#include <cstdlib>
#include "test.hpp"
#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>

/***********************************************************************
  Function Definitions
***********************************************************************/


// test_complex_subview -- Test real() and imag() subviews of a complex
//                         matrix.
void
test_complex_subview()
{
   using namespace vsip;

   length_type const N0 = 7;
   length_type const N1 = 9;

   typedef Matrix<cscalar_f>::realview_type
		realview_type;
   typedef Matrix<cscalar_f>::imagview_type
		imagview_type;

   Matrix<cscalar_f>
		cv(N0, N1, cscalar_f(1.f, 2.f));

   realview_type	rv(cv.real());
   imagview_type	iv(cv.imag());

   for (index_type i0=0; i0<N0; ++i0) {
      for (index_type i1=0; i1<N1; ++i1) {
	 insist(equal(rv.get(i0, i1), 1.f));
	 insist(equal(iv.get(i0, i1), 2.f));

	 rv.put(i0, i1,  3.f);
	 iv.put(i0, i1, -1.f);

	 insist(equal(cv.get(i0, i1), cscalar_f(3.f, -1.f)));
	 }
      }
}



int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);

  test_complex_subview();

  // Test Matrix<scalar_f>.

  vsip::Matrix<> matrix_scalar_f (7, 3, 3.4);
  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.get (i, j);

  // Test Matrix<scalar_i>.

  vsip::Matrix<vsip::scalar_i>
		matrix_scalar_i (7, 3, 3);
  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_i.get (i, j);

  // Copy Matrixes.

  vsip::Matrix<>
		v (matrix_scalar_f.block ());
  vsip::Matrix<>
		w (v);
  vsip::Matrix<>
		x (3, 3);
  vsip::Matrix<>
		y (v.size (0), v.size (1));
  y = matrix_scalar_f;
  insist (equal (y.size (), y.size (0) * y.size (1)));
  insist (equal (y.get (2, 2), static_cast<vsip::scalar_f>(3.4)));

  x = v (vsip::Domain<2>(3, 3));
  insist (equal (x.size (), x.size (0) * x.size (1)));
  insist (equal (x.get (2, 2), static_cast<vsip::scalar_f>(3.4)));

  // Test ordinary subviews of Matrixes.

  vsip::Domain<2>	dom ((vsip::Domain<1>(3)),
			     (vsip::Domain<1>(3)));
  insist (x.get (dom).size (0) == 3 && x.get (dom).size (0) == 3 &&
	  x.get (dom).size () == 3 * 3);
  insist (equal (x.get (dom).get (2, 2), static_cast<vsip::scalar_f>(3.4))
	  && equal (x.get (2, 2), static_cast<vsip::scalar_f>(3.4)));
  vsip::Domain<2>	domTwo ((vsip::Domain<1>(2)),
				(vsip::Domain<1>(3)));
  insist (x.get (domTwo).size (0) == 2);
  insist (x.get (domTwo).size (1) == 3);
  insist (x.get (domTwo).size ()  == 2*3);
  insist (equal (x.get (domTwo).get (1, 0), static_cast<vsip::scalar_f>(3.4))
	  && equal (x.get (1, 0), static_cast<vsip::scalar_f>(3.4)));

  vsip::Matrix<>::subview_type
		xsv (x (domTwo));
  x (domTwo).put (1, 2, 2.4);
  insist (equal (x.get (domTwo).get (1, 2), static_cast<vsip::scalar_f>(2.4))
	  && equal (x.get (1, 2), static_cast<vsip::scalar_f>(2.4)));
  insist (equal (x (domTwo).get (1, 2), static_cast<vsip::scalar_f>(2.4)));

  x (domTwo).put (1, 2, 3.4);
  insist (equal (x.get (domTwo).get (1, 2), static_cast<vsip::scalar_f>(3.4))
	  && equal (x.get (1, 2), static_cast<vsip::scalar_f>(3.4)));
  insist (equal (x (domTwo).get (1, 2), static_cast<vsip::scalar_f>(3.4)));

  x (dom)(domTwo).put(0, 1, static_cast<vsip::scalar_f>(2.1));
  insist (equal (x.get (dom)(domTwo).get (0, 1),
		 static_cast<vsip::scalar_f>(2.1)) && 
	  equal (x.get (dom).get (0, 1), static_cast<vsip::scalar_f>(2.1)));

  // Test column subviews.

  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.put(i, j, static_cast<vsip::scalar_f>(3.4));

  matrix_scalar_f.col (1).put (2, static_cast<vsip::scalar_f>(3.0));
  insist (equal (matrix_scalar_f.col (1).get (2),
		 static_cast<vsip::scalar_f>(3.0)));
  insist (equal (matrix_scalar_f.get (2, 1),
		 static_cast<vsip::scalar_f>(3.0)));
  matrix_scalar_f.col (1).put (1, 2.0);
  insist (equal (matrix_scalar_f.col (1).get (1),
		 static_cast<vsip::scalar_f>(2.0)));
  insist (equal (matrix_scalar_f.get (1, 1),
		 static_cast<vsip::scalar_f>(2.0)));
  insist (equal (matrix_scalar_f.col (1).size (),
		 static_cast<vsip::length_type>(7)));
  insist (equal (matrix_scalar_f.col (1).get (vsip::Domain<1>(4)).get (2),
		 static_cast<vsip::scalar_f>(3.0)));

  // Test row subviews.

  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.put(i, j, 3.4);

  matrix_scalar_f.row (1).put (2, static_cast<vsip::scalar_f>(3.0));
  insist (equal (matrix_scalar_f.row (1).get (2),
		 static_cast<vsip::scalar_f>(3.0)));
  insist (equal (matrix_scalar_f.get (1, 2),
		 static_cast<vsip::scalar_f>(3.0)));
  matrix_scalar_f.row (1).put (1, static_cast<vsip::scalar_f>(2.0));
  insist (equal (matrix_scalar_f.row (1).get (1),
		 static_cast<vsip::scalar_f>(2.0)));
  insist (equal (matrix_scalar_f.get (1, 1), 
		 static_cast<vsip::scalar_f>(2.0)));

  // Test diagonal subviews.

  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.put(i, j, 3.4);

  matrix_scalar_f.diag ().put (2, static_cast<vsip::scalar_f>(3.0));
  insist (equal (matrix_scalar_f.diag (0).get (2),
		 static_cast<vsip::scalar_f>(3.0)));
  insist (equal (matrix_scalar_f.get (2, 2),
		 static_cast<vsip::scalar_f>(3.0)));
  matrix_scalar_f.put (1, 1, static_cast<vsip::scalar_f>(2.0));
  insist (equal (matrix_scalar_f.diag (0).get (1),
		 static_cast<vsip::scalar_f>(2.0)));
  insist (equal (matrix_scalar_f.get (1, 1),
		 static_cast<vsip::scalar_f>(2.0)));

  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.put(i, j, 3.4);

  matrix_scalar_f.diag (1).put (1, static_cast<vsip::scalar_f>(3.0));
  insist (equal (matrix_scalar_f.diag (1).get (1), 
		 static_cast<vsip::scalar_f>(3.0)));
  insist (equal (matrix_scalar_f.get (1, 2),
		 static_cast<vsip::scalar_f>(3.0)));
  matrix_scalar_f.diag (1).put (0, static_cast<vsip::scalar_f>(2.0));
  insist (equal (matrix_scalar_f.diag (1).get (0),
		 static_cast<vsip::scalar_f>(2.0)));
  insist (equal (matrix_scalar_f.get (0, 1),
		 static_cast<vsip::scalar_f>(2.0)));

  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.put(i, j, 3.4);

  matrix_scalar_f.diag (-1).put (1, static_cast<vsip::scalar_f>(3.0));
  insist (equal (matrix_scalar_f.diag (-1).get (1),
		 static_cast<vsip::scalar_f>(3.0)));
  insist (equal (matrix_scalar_f.get (2, 1),
		 static_cast<vsip::scalar_f>(3.0)));
  matrix_scalar_f.diag (-1).put (2, static_cast<vsip::scalar_f>(2.0));
  insist (equal (matrix_scalar_f.diag (-1).get (2),
		 static_cast<vsip::scalar_f>(2.0)));
  insist (equal (matrix_scalar_f.get (3, 2),
		 static_cast<vsip::scalar_f>(2.0)));

  // Test transpose subviews.

  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.put(i, j, 3.4);

  matrix_scalar_f.transpose ().put (2, 5, static_cast<vsip::scalar_f>(3.0));
  insist (equal (matrix_scalar_f.transpose ().get (2, 5),
		 static_cast<vsip::scalar_f>(3.0)));
  insist (equal (matrix_scalar_f.get (5, 2),
		 static_cast<vsip::scalar_f>(3.0)));
  matrix_scalar_f.transpose ().put(2, 1, static_cast<vsip::scalar_f>(2.0));
  insist (equal (matrix_scalar_f.transpose ().get (2, 1),
		 static_cast<vsip::scalar_f>(2.0)));
  insist (equal (matrix_scalar_f.get(1, 2),
		 static_cast<vsip::scalar_f>(2.0)));
  insist (equal (matrix_scalar_f.transpose ().transpose ().get(1, 2),
		 static_cast<vsip::scalar_f>(2.0)));



  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.put(i, j, 100.f*i + j);
  

  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
       insist (equal (matrix_scalar_f.transpose().get(j, i),
		      100.f*i + j));

  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
       insist (equal (matrix_scalar_f.transpose().transpose().get(i, j),
		      100.f*i + j));



  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.transpose().put(j, i, 100.f*i + j);

  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
       insist (equal (matrix_scalar_f.get(i, j),
		      100.f*i + j));



  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.put(i, j, 3.4);

  matrix_scalar_f.transpose ()
    .row (1).put (4, static_cast<vsip::scalar_f>(3.0));
  insist (equal (matrix_scalar_f.transpose ().row (1).get (4),
		 static_cast<vsip::scalar_f>(3.0)));
  insist (equal (matrix_scalar_f.get (4, 1), 
		 static_cast<vsip::scalar_f>(3.0)));

  // Construct a few Matrixes from other Matrixes.

  for (vsip::index_type i = 0; i < 7; ++i)
    for (vsip::index_type j = 0; j < 3; ++j)
      matrix_scalar_f.put(i, j, 3.4);
  vsip::Matrix<vsip::scalar_f>
		matrix_scalar_f_copy
    (matrix_scalar_f (vsip::Domain<2>(3, 3)));
  insist (equal (matrix_scalar_f_copy.size (0),
		 static_cast<vsip::length_type>(3)));
  insist (equal (matrix_scalar_f_copy.size (1),
		 static_cast<vsip::length_type>(3)));
  insist (equal (matrix_scalar_f_copy.get (0, 0),
		 static_cast<vsip::scalar_f>(3.4)));
  // Test that reference semantics are not used.
  matrix_scalar_f.put (0, 0, static_cast<vsip::scalar_f>(4.0));
  insist (equal (matrix_scalar_f_copy.get (0, 0),
		 static_cast<vsip::scalar_f>(3.4)));
  insist (equal (matrix_scalar_f.get (0, 0),
		 static_cast<vsip::scalar_f>(4.0)));
  matrix_scalar_f_copy.put (0, 1, static_cast<vsip::scalar_f>(4.0));
  insist (equal (matrix_scalar_f_copy.get (0, 1),
		 static_cast<vsip::scalar_f>(4.0)));
  insist (equal (matrix_scalar_f.get (0, 1),
		 static_cast<vsip::scalar_f>(3.4)));

  return EXIT_SUCCESS;
}

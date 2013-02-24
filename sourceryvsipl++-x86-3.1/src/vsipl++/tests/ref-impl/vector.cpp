/***********************************************************************

  File:   vector.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   07/05/2002

  Contents: Very, very simple tests of the Vector view class.

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
#include <vsip/domain.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>

/***********************************************************************
  Function Definitions
***********************************************************************/

// test_real_subview -- Test real() subviews of a complex vector.
void
test_real_subview()
{
   using namespace vsip;

   length_type const
		N = 7;

   typedef Vector<cscalar_f>::realview_type
		realview_type;

   Vector<cscalar_f>
		cvec(N, cscalar_f(1.f, 2.f));

   realview_type	rvec(cvec.real());

   for (index_type i=0; i<N; ++i) {
      insist(equal(rvec.get(i), 1.f));

      rvec.put(i, 3.f);

      insist(equal(cvec.get(i), cscalar_f(3.f, 2.f)));
      }
}



// test_imag_subview -- Test imaginary() subviews of a complex vector.
void
test_imag_subview()
{
   using namespace vsip;

   length_type const
		N = 7;

   typedef Vector<cscalar_f>::imagview_type
		imagview_type;

   Vector<cscalar_f>
		cvec(N, cscalar_f(1.f, 2.f));

   imagview_type	ivec(cvec.imag());

   for (index_type i=0; i<N; ++i) {
      insist(equal(ivec.get(i), 2.f));

      ivec.put(i, 3.f);

      insist(equal(cvec.get(i), cscalar_f(1.f, 3.f)));
      }
}


int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);

  test_real_subview();
  test_imag_subview();

  /* Test Vector<scalar_f>.  */

  vsip::Vector<> vector_scalar_f (7, 3.4);
  for (vsip::index_type i = 0; i < 7; ++i)
    insist(equal(vector_scalar_f.get(i), 3.4f));

  /* Test Vector<scalar_i>.  */

  vsip::Vector<vsip::scalar_i>
		vector_int (7, 3);
  for (vsip::index_type i = 0; i < 7; ++i)
    insist(equal(vector_int.get(i), 3));

  /* Test Vector<bool>.  */

  vsip::Vector<bool>
		vector_bool (7, true);
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (vector_bool.get(i), true));

  /* Test Vector<index_type>.  */

  vsip::Vector<vsip::index_type>
		vector_index_one (7, 3);
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (vector_index_one.get(i),
		   static_cast<vsip::index_type>(3)));
  vsip::Vector<vsip::index_type>
		vector_index_two (7, 4);
  vector_index_two = vector_index_one;
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (vector_index_two.get(i),
		   static_cast<vsip::index_type>(3)));

  /* Test Vector<Index<2>>.  */

  vsip::Vector<vsip::Index<2> >
		vector_index2_one (7, vsip::Index<2>(1,2));
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (vector_index2_one.get(i), vsip::Index<2>(1,2)));
  vsip::Vector<vsip::Index<2> >
		vector_index2_two (7, vsip::Index<2>(2,1));
  vector_index2_two = vector_index2_one;
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (vector_index2_two.get(i), vsip::Index<2>(1,2)));

  /* Copy Vectors.  */

  vsip::Vector<>
		v (vector_scalar_f.block ());
  vsip::Vector<>
		w (v);
  vsip::Vector<>
		x (3);
  vsip::Vector<>
		y (v.size ());
  y = vector_scalar_f;
  insist (equal (y.get (2), static_cast<vsip::scalar_f>(3.4)));

  x = v (vsip::Domain<1>(3));
  insist (x.size () == x.size (0));
  insist (x.size () == 3);
  insist (x.size () == x.length ());
  insist (equal (x.get (2), static_cast<vsip::scalar_f>(3.4)));

  /* Test subviews of Vectors.  */

  vsip::Domain<1>	dom (3);
  insist (x.get (dom).size () == 3);
  insist (equal (x.get (dom).get (2), static_cast<vsip::scalar_f>(3.4)) && 
	  equal (x.get (2), static_cast<vsip::scalar_f>(3.4)));
  vsip::Domain<1>	domTwo (2);
  vsip::Vector<>::const_subview_type
		xsv (x.get (domTwo));
  insist (x.get (domTwo).size () == 2);
  insist (xsv.size () == 2);
  insist (equal (x.get (domTwo).get (1), static_cast<vsip::scalar_f>(3.4)) &&
	  equal (x.get (1), static_cast<vsip::scalar_f>(3.4)));

  x (domTwo).put (1, static_cast<vsip::scalar_f>(2.4));
  insist (equal (x.get (domTwo).get (1), static_cast<vsip::scalar_f>(2.4)) && 
	  equal (x.get (1), static_cast<vsip::scalar_f>(2.4)));
  insist (equal (x (domTwo).get (1), static_cast<vsip::scalar_f>(2.4)) &&
	  equal (x.get (1), static_cast<vsip::scalar_f>(2.4)));
  
  x (domTwo).put(1, static_cast<vsip::scalar_f>(3.4));
  insist (equal (x.get (domTwo).get (1), static_cast<vsip::scalar_f>(3.4)) &&
	  equal (x.get (1), static_cast<vsip::scalar_f>(3.4)));
  insist (equal (x (domTwo).get (1), static_cast<vsip::scalar_f>(3.4)) &&
	  equal (x.get (1), static_cast<vsip::scalar_f>(3.4)));

  x (dom)(domTwo).put(0, static_cast<vsip::scalar_f>(2.1));
  insist (equal (x.get (dom)(domTwo).get (0),
		 static_cast<vsip::scalar_f>(2.1)) && 
	  equal (x.get (dom).get (0), static_cast<vsip::scalar_f>(2.1)) &&
	  equal (x.get (0), static_cast<vsip::scalar_f>(2.1)));

  /* Test a few assignment operators.  */
  for (vsip::index_type i = 0; i < w.size (); ++i)
    w.put(i, 3.4);
  vsip::Vector<> vector_scalar_fTwo(7, 2.3);
  w += vector_scalar_fTwo;
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (w.get (i), static_cast<vsip::scalar_f>(5.7)));
  w += w += vector_scalar_fTwo;
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (w.get (i), static_cast<vsip::scalar_f>(16.0)));
  w -= w;
  insist (equal (w.get (2), static_cast<vsip::scalar_f>(0.0)));
  for (vsip::index_type i = 0; i < w.size (); ++i)
    w.put(i, 3.4);
  w -= 1.4;
  insist (equal (w.get (2), static_cast<vsip::scalar_f>(2.0)));
  for (vsip::index_type i = 0; i < w.size (); ++i)
    w.put(i, 3.4);
  w /= static_cast<vsip::scalar_f>(2.0);
  insist (equal (w.get (2), static_cast<vsip::scalar_f>(1.7)));

  /* Construct a few Vectors from other Vectors.  */

  vsip::Vector<> copy_vector_int (vector_int);
  insist (equal (copy_vector_int.get (0),
		 static_cast<vsip::scalar_f>(3.0)));
  /* Test that reference semantics are not used.  */
  copy_vector_int.put (0, static_cast<vsip::scalar_f>(4.0));
  insist (equal (copy_vector_int.get (0),
		 static_cast<vsip::scalar_f>(4.0)));
  insist (equal (vector_int.get (0), 3));
  vsip::Vector<> copy_vector_subview (x.get (domTwo));

  return EXIT_SUCCESS;
}

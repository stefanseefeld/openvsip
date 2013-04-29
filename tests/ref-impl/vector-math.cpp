/***********************************************************************

  File:   vector-math.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   07/05/2002

  Contents: Very, very simple tests of element-wise Vector extensions.

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
#include <vsip/complex.hpp>
#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include "test-util.hpp"

/***********************************************************************
  Function Definitions
***********************************************************************/



// -------------------------------------------------------------------- //
int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);
  vsip::Vector<vsip::scalar_f>
		vector_scalarf (7, 3.4);
  vsip::Vector<vsip::cscalar_f>
		vector_cscalarf (7, vsip::cscalar_f (3.3, 3.3));
  vsip::Vector<vsip::scalar_i>
		vector_scalari (7, 3);
#ifdef XBOOL
  vsip::Vector<bool>
		vector_bool (7, false);
#endif

  /* Test assignment.  */
  vector_scalarf = 2.2;
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(2.2));

  /* Test assignment of one vector's values to another.  */
  vsip::Vector<> vector_lhs (7, 0.0);
  check_entry (vector_lhs, 2, static_cast<vsip::scalar_f>(0.0));
  vector_lhs = vector_scalarf;
  check_entry (vector_lhs, 2, static_cast<vsip::scalar_f>(2.2));

  /* Test assignment of a scalar to a vector.  */
  vector_scalarf = 0.0;
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(0.0));

  /* Test arccosine.  This should yield a vector of 1.0's.  */
  vector_scalarf = 1.0;
  vector_scalarf = vsip::acos (vector_scalarf);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(0.0));

  /* Test add.  */
  vector_scalarf = 2.1;
  vector_scalarf = vsip::add (vector_scalarf, vector_scalarf);
  check_entry (vector_scalarf, 3, static_cast<vsip::scalar_f>(4.2));

  /* Test am.  */
  vector_scalarf = 3.0;
  vector_scalarf = am (vector_scalarf, vector_scalarf, vector_scalarf);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(18.0));

  /* Test arg.  */
  vector_cscalarf = vsip::cscalar_f (3.3, 0.0);
  vector_scalarf = vsip::arg (vector_cscalarf);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(0.0));

  /* Test ceil.  */
  vector_scalarf = 2.1;
  vector_scalarf = vsip::ceil (vector_scalarf);
  check_entry (vector_scalarf, 3, static_cast<vsip::scalar_f>(3.0));

#ifdef XBOOL
  /* Test eq.  */
  vector_bool = vsip::eq (vector_scalari, vector_scalari);
  check_entry (vector_bool, 2, true);
  check_entry (vsip::eq (vector_scalari, vector_scalari), 2, true);
  vector_bool = vsip::eq (vector_scalarf, vector_scalarf);
  check_entry (vector_bool, 2, true);
#if 0 /* tvcpp0p8 does not define the underlying function.  */
  insist (vsip::eq (vector_cscalarf, vector_cscalarf));
#endif /* tvcpp0p8 does not define the underlying function.  */
#endif

  /* Test euler.  */
  vector_scalarf = 0.0;
  vector_cscalarf = vsip::euler (vector_scalarf);
  check_entry (vector_cscalarf, 2, vsip::cscalar_f (1.0));

  /* Test fmod.  */
  vector_scalarf = 3.4;
  check_entry (vsip::fmod (vector_scalarf, vector_scalarf), 3,
	       static_cast<vsip::scalar_f>(0.0));

  /* Test imag.  */
  vector_cscalarf = vsip::cscalar_f (3.3, 3.2);
  vector_scalarf = vsip::imag (vector_cscalarf);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(3.2));

  /* Test magsq.  */
  vector_cscalarf = vsip::cscalar_f (3.3, 0.0);
  vector_scalarf = vsip::magsq (vector_cscalarf);
  check_entry (vector_scalarf, 2,
	       static_cast<vsip::scalar_f>(3.3) *
	       static_cast<vsip::scalar_f>(3.3));
  vector_cscalarf = vsip::cscalar_f (0.0, 3.3);
  vector_scalarf = vsip::magsq (vector_cscalarf);
  check_entry (vector_scalarf, 2,
	       static_cast<vsip::scalar_f>(3.3) *
	       static_cast<vsip::scalar_f>(3.3));

  /* Test ne and !=.  */
  check_entry (ne (vector_scalarf, vector_scalarf),
	       2,
	       false);
  check_entry (vector_scalarf != vector_scalarf,
	       2,
	       false);
#if 0 /* tvcpp0p8 does not define the underlying function.  */
  check_entry (ne (vector_cscalarf, vector_cscalarf),
	       2,
	       false);
#endif /* tvcpp0p8 does not define the underlying function.  */

  /* Test ne and -.  */
  vector_scalarf = 3.4;
  check_entry (neg (vector_scalarf),
	       2,
	       static_cast<vsip::scalar_f>(-3.4));
  check_entry (-vector_scalarf,
	       2,
	       static_cast<vsip::scalar_f>(-3.4));

#ifdef XBOOL
  /* Test lnot.  */
  vector_bool.put (3, true);
  vector_bool = vsip::lnot (vector_bool);
  check_entry (vector_bool, 3, false);
#endif

  /* Test sub.  */
  vector_scalarf = 3.4;
  check_entry (vector_scalarf - vector_scalarf, 2,
	       static_cast<vsip::scalar_f>(0.0));

  /* Test adding a scalar to a vector.  */
  vector_scalarf = 3.4;
  check_entry (vsip::add (static_cast<vsip::scalar_f>(-3.4), vector_scalarf),
	       2,
	       static_cast<vsip::scalar_f>(0.0));

  /* Test am with vector, scalar, and vector.  */
  vector_scalarf = 1.0;
  check_entry (vsip::am (vector_scalarf,
			 static_cast<vsip::scalar_f>(-3.4),
			 vector_scalarf),
	       2,
	       static_cast<vsip::scalar_f>(-2.4));
  vector_cscalarf = vsip::cscalar_f (0.0, 0.0);
  check_entry (vsip::am (vector_cscalarf,
			 vsip::cscalar_f(-3.4, -3.4),
			 vector_cscalarf),
	       2,
	       vsip::cscalar_f(0.0, 0.0));

  /* Test ma with vector, scalar, scalar.  */
  vector_scalarf = 1.0;
  vector_scalarf = vsip::ma (vector_scalarf,
			     static_cast<vsip::scalar_f>(-1.0),
			     static_cast<vsip::scalar_f>(2.0));
  check_entry (vsip::ma (vector_scalarf,
			 static_cast<vsip::scalar_f>(-1.0),
			 static_cast<vsip::scalar_f>(2.0)),
	       2,
	       static_cast<vsip::scalar_f>(1.0));

  /* Test expoavg.  */
  vector_scalarf = 1.0;
  check_entry (vsip::expoavg (static_cast<vsip::scalar_f>(0.5),
			      vector_scalarf, vector_scalarf),
	       2,
	       static_cast<vsip::scalar_f>(1.0));
  

  /* Test arithmetic on vectors.  */
  check_entry (static_cast<vsip::scalar_f>(0.5) + vector_scalarf,
	       2,
	       static_cast<vsip::scalar_f>(1.5));

  vector_scalarf = 1.1;
  vector_scalarf = static_cast<vsip::scalar_f>(2.0) * vector_scalarf
    + vector_scalarf;
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(3.3));
  vector_scalarf = 1.1;
  vector_scalarf = static_cast<vsip::scalar_f>(2.0) * vector_scalarf
    + (static_cast<vsip::scalar_f>(4.0) + vector_scalarf);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(7.3));

  /* Test incrementing a vector.  */
  vector_scalarf = 0.0;
  vector_scalarf += static_cast<vsip::scalar_f>(1.1);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(1.1));

  /* Test adding two vectors of complex numbers.  */
  vsip::Vector<vsip::cscalar_f>
		vector_complex_a (4, vsip::cscalar_f(1.0, 1.0));
  vector_complex_a += vector_complex_a;
  check_entry (vector_complex_a, 2, vsip::cscalar_f(2.0, 2.0));
  vsip::Vector<vsip::cscalar_f>
		vector_complex_b (4, vsip::cscalar_f(1.0, 1.0));
  vector_complex_b += vector_complex_a;
  check_entry (vector_complex_b, 2, vsip::cscalar_f(3.0, 3.0));

#if 0 /* The VSIPL++ specification does not require supporting
	 addition of vectors with different value types except for a
	 few special cases.  */
  /* Test addition of vector with complex numbers and non-complex
     numbers.  */
  vsip::Vector<vsip::scalar_i>
		vector_int (4, -17);
  vector_complex_b = vector_complex_b + vector_int;
  check_entry (vector_complex_b, 2, vsip::cscalar_f(-14.0, 3.0));
  vector_complex_b += static_cast<vsip::scalar_f>(-13.3);
  check_entry (vector_complex_b, 2, vsip::cscalar_f(-0.7, 3.0));
#endif

  return EXIT_SUCCESS;
}

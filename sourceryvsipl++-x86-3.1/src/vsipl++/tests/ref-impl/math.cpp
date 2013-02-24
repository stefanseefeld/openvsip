/***********************************************************************

  File:   math.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   07/02/2002

  Contents: Very, very simple tests of scalar math functions.

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
#include <vsip/math.hpp>
#include <vsip/support.hpp>

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);

  /* Test acos.  */
  insist (equal (vsip::acos (static_cast<vsip::scalar_f>(1.0)),
		 static_cast<vsip::scalar_f>(0.0)));

  /* Test add.  */
  insist (equal (vsip::add (static_cast<vsip::scalar_f>(1.2),
			    static_cast<vsip::scalar_f>(1.2)),
		 static_cast<vsip::scalar_f> (2.4)));
  insist (equal (vsip::add (static_cast<vsip::scalar_f>(1.2),
			    vsip::cscalar_f (1.2)),
		 static_cast<vsip::cscalar_f>(2.4)));
  insist (equal (static_cast<vsip::scalar_f>(1.2) + 
			    vsip::cscalar_f (1.2),
		 static_cast<vsip::cscalar_f>(2.4)));

  /* Test arg.  */
  insist (equal (vsip::arg (vsip::cscalar_f (1.2, 0.0)),
		 static_cast<vsip::scalar_f>(0.0)));

  /* Test ceil.  */
  insist (equal (vsip::ceil (static_cast<vsip::scalar_f>(2.3)),
		 static_cast<vsip::scalar_f>(3.0)));

  /* Test cos.  */
  insist (equal (vsip::cos (static_cast<vsip::scalar_f>(0.0)),
		 static_cast<vsip::scalar_f>(1.0)));
  
  /* Test eq.  */
  insist (equal (vsip::eq (static_cast<vsip::scalar_f>(0.0),
			   static_cast<vsip::scalar_f>(1.0)),
		 false));

  /* Test exp.  */
  insist (equal (vsip::exp (static_cast<vsip::scalar_f>(0.0)),
		 static_cast<vsip::scalar_f>(1.0)));

  /* Test ge.  */
  insist (equal (vsip::ge (static_cast<vsip::scalar_f>(0.0),
			   static_cast<vsip::scalar_f>(1.0)),
		 false));
  insist (equal (static_cast<vsip::scalar_f>(0.0) >= 
		 static_cast<vsip::scalar_f>(1.0),
		 false));

  /* Test jmul.  */
  insist (equal (vsip::jmul (vsip::cscalar_f (1.0, 2.0),
			     vsip::cscalar_f (3.0, 4.0)),
		 vsip::cscalar_f (11.0, 2.0)));

  /* Test ma.  */
  // insist (equal (vsip::ma (-17, -20, 3), 343)); // NRBS

  /* Test max.  */
  insist (equal (vsip::max (static_cast<vsip::scalar_f>(-17.0),
			    static_cast<vsip::scalar_f>(-20.0)),
		 static_cast<vsip::scalar_f>(-17.0)));

  /* Test maxmg.  */
  insist (equal (vsip::maxmg (static_cast<vsip::scalar_f>(-17.0),
			      static_cast<vsip::scalar_f>(-20.0)),
		 static_cast<vsip::scalar_f>(20.0)));

  /* Test mul.  */
  insist (equal (vsip::mul (static_cast<vsip::scalar_f>(2.0),
			    static_cast<vsip::scalar_f>(1.2)),
		 static_cast<vsip::scalar_f> (2.4)));
  insist (equal (vsip::mul (static_cast<vsip::scalar_f>(2.0),
			    vsip::cscalar_f (1.2)),
		 static_cast<vsip::cscalar_f>(2.4)));
  insist (equal (static_cast<vsip::scalar_f>(2.0) * 
			    vsip::cscalar_f (1.2),
		 static_cast<vsip::cscalar_f>(2.4)));

  /* Test neg.  */
  insist (equal (vsip::neg (static_cast<vsip::scalar_f>(2.0)),
		 static_cast<vsip::scalar_f>(-2.0)));

  /* Test pow.  */
  insist (equal (vsip::pow (static_cast<vsip::scalar_f>(2.0),
			    static_cast<vsip::scalar_f>(3.0)),
		 static_cast<vsip::scalar_f>(8.0)));

  /* Test recip.  */
  insist (equal (vsip::recip (static_cast<vsip::scalar_f>(2.0)),
		 static_cast<vsip::scalar_f>(0.5)));
  insist (equal (vsip::recip (static_cast<vsip::scalar_f>(4.0)),
		 static_cast<vsip::scalar_f>(0.25)));

  /* Test rsqrt.  */
  insist (equal (vsip::rsqrt (static_cast<vsip::scalar_f>(4.0)),
		 static_cast<vsip::scalar_f>(0.5)));

  /* Test sub.  */
  insist (equal (vsip::sub (vsip::cscalar_f (1.0, 2.0),
			     vsip::cscalar_f (3.0, 4.0)),
		 vsip::cscalar_f (-2.0, -2.0)));

  return EXIT_SUCCESS;
}

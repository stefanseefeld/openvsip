/***********************************************************************

  File:   complex.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   08/12/2002

  Contents: Very simple tests of the vsip::complex class.

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
  Notes
***********************************************************************/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cstdlib>
#include "test.hpp"
#include <vsip/complex.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>

/***********************************************************************
  Macros
***********************************************************************/

/***********************************************************************
  Forward Declarations
***********************************************************************/

/***********************************************************************
  Type Declarations
***********************************************************************/

/***********************************************************************
  Class Declarations
***********************************************************************/

/***********************************************************************
  Variable Declarations
***********************************************************************/

/***********************************************************************
  Function Declarations
***********************************************************************/

template <typename T>
inline
bool
equal (vsip::complex<T> operand,
       T                real_part,
       T                imag_part);
			/* Requires:
			     T must be a floating point type.
			   Returns:
			     true iff the OPERAND's real part equals
			     REAL_PART and its imaginary part equals
			     IMAG_PART.  */

/***********************************************************************
  Variable Definitions
***********************************************************************/

/***********************************************************************
  Inline Function Definitions
***********************************************************************/

template <typename T>
inline
bool
equal (vsip::complex<T> operand,
       T                real_part,
       T                imag_part)
{
  return equal (operand.real (), real_part) &&
    equal (operand.imag (), imag_part);
}

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);

  // Test the creation of three complex numbers.
  vsip::cscalar_f a (static_cast<vsip::scalar_f>(2.0));
  vsip::cscalar_f b (static_cast<vsip::scalar_f>(2),
		     static_cast<vsip::scalar_f>(3));
  vsip::cscalar_f c (static_cast<vsip::scalar_f>(4.0));

  insist (equal (a.real (),
		 static_cast<vsip::scalar_f>(2.0)) &&
	  equal (a.imag (),
		 static_cast<vsip::scalar_f>(0.0)));
  insist (equal (b.real (),
		 static_cast<vsip::scalar_f>(2.0)) &&
	  equal (b.imag (),
		 static_cast<vsip::scalar_f>(3.0)));
  insist (equal (c.real (),
		 static_cast<vsip::scalar_f>(4.0)) && 
	  equal (c.imag (),
		 static_cast<vsip::scalar_f>(0.0)));

  // Test some arithmetic operations.
  insist (equal (static_cast<vsip::scalar_f>(2.0) * a, 
		 static_cast<vsip::scalar_f>(4.0),
		 static_cast<vsip::scalar_f>(0.0)));
  insist (equal (a + a,
		 static_cast<vsip::scalar_f>(4.0),
		 static_cast<vsip::scalar_f>(0.0)));
  insist (a == a);
  insist (vsip::operator==(a,a));
  insist (equal (vsip::real (b),
		 static_cast<vsip::scalar_f>(2.0)));
  insist (equal (vsip::abs (b) * vsip::abs (b),
		 static_cast<vsip::scalar_f>(13.0)));
  return EXIT_SUCCESS;
}

/***********************************************************************

  File:   matrix-math.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   08/12/2002

  Contents: Very, very simple tests of element-wise Matrix extensions.

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
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>
#include <vsip/support.hpp>

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);
  vsip::Matrix<>
		matrix_scalarf (7, 3, 3.4); 
  check_entry (matrix_scalarf, 1, 1, static_cast<vsip::scalar_f>(3.4));
  vsip::Matrix<vsip::cscalar_f>
		matrix_cscalarf (7, 3, vsip::cscalar_f (3.3, 3.3));
  vsip::Matrix<vsip::scalar_i>
		matrix_scalari (7, 3, 4);

  // Test assignment.
  matrix_scalarf = 2.2;
  check_entry (matrix_scalarf, 1, 1, static_cast<vsip::scalar_f>(2.2));

  // Test assignment of one matrix's values to another.
  vsip::Matrix<> matrix_lhs(7, 3, 0.0);
  check_entry (matrix_lhs, 1, 1, static_cast<vsip::scalar_f>(0.0));
  matrix_lhs = matrix_scalarf;
  check_entry (matrix_lhs, 1, 1, static_cast<vsip::scalar_f>(2.2));

  // Test assignment of a scalar to a matrix.
  matrix_scalarf = 0.0;
  check_entry (matrix_scalarf, 1, 1, static_cast<vsip::scalar_f>(0.0));

  // Test arccosine.  This should yield a matrix of 1.0's.
  matrix_scalarf = 1.0;
  matrix_scalarf = vsip::acos (matrix_scalarf);
  check_entry (matrix_scalarf, 1, 1, static_cast<vsip::scalar_f>(0.0));

  // Test add.
  matrix_scalarf = 2.1;
  matrix_scalarf = vsip::add (matrix_scalarf, matrix_scalarf);
  check_entry (matrix_scalarf, 2, 1, static_cast<vsip::scalar_f>(4.2));
  matrix_cscalarf = vsip::add (matrix_scalarf, matrix_cscalarf);
  check_entry (matrix_cscalarf, 2, 1, vsip::cscalar_f(7.5, 3.3));
  matrix_cscalarf = vsip::add (matrix_cscalarf, matrix_cscalarf);
  check_entry (matrix_cscalarf, 2, 1, vsip::cscalar_f(15.0, 6.6));
#if 0 /* tvcpp0p8 does not define vsip_madd_i.  */
  matrix_scalari = vsip::add (matrix_scalari, matrix_scalari);
  check_entry (matrix_scalari, 2, 1, 8);
#endif /* tvcpp0p8 does not define vsip_madd_i.  */

  // Test arg.
  matrix_cscalarf = vsip::cscalar_f (3.3, 0.0);
  matrix_scalarf = vsip::arg (matrix_cscalarf);
  check_entry (matrix_scalarf, 2, 1, static_cast<vsip::scalar_f>(0.0));

  // Test ceil.
  matrix_scalarf = 2.1;
  matrix_scalarf = vsip::ceil (matrix_scalarf);
  check_entry (matrix_scalarf, 3, 1, static_cast<vsip::scalar_f>(3.0));

  // Test cosine.  This should yield a matrix of 1.0's.
  matrix_scalarf = 0.0;
  matrix_scalarf = vsip::cos (matrix_scalarf);
  check_entry (matrix_scalarf, 1, 1, static_cast<vsip::scalar_f>(1.0));

  // Test euler.
  matrix_scalarf = 0.0;
  matrix_cscalarf = vsip::euler (matrix_scalarf);
  check_entry (matrix_cscalarf, 1, 1, vsip::cscalar_f (1.0));

  // Test exponentiation.  This should yield a matrix of 1.0's.
  matrix_scalarf = 0.0;
  matrix_scalarf = vsip::exp (matrix_scalarf);
  check_entry (matrix_scalarf, 1, 1, static_cast<vsip::scalar_f>(1.0));

  // Test fmod.
  matrix_scalarf = 3.4;
  check_entry (vsip::fmod (matrix_scalarf, matrix_scalarf), 3, 2,
	       static_cast<vsip::scalar_f>(0.0));

  // Test mag.
  matrix_cscalarf = vsip::cscalar_f (3.3, 0.0);
  matrix_scalarf = vsip::mag (matrix_cscalarf);
  check_entry (matrix_scalarf, 2, 1, static_cast<vsip::scalar_f>(3.3));
  matrix_scalarf = -3.2;
  matrix_scalarf = vsip::mag (matrix_scalarf);
  check_entry (matrix_scalarf, 2, 1, static_cast<vsip::scalar_f>(3.2));

  // Test real.
  matrix_cscalarf = vsip::cscalar_f (3.3, 0.0);
  matrix_scalarf = vsip::real (matrix_cscalarf);
  check_entry (matrix_scalarf, 2, 1, static_cast<vsip::scalar_f>(3.3));

  // Test sqrt.
  matrix_scalarf = 4.0;
  matrix_scalarf = vsip::sqrt (matrix_scalarf);
  check_entry (matrix_scalarf, 2, 1, static_cast<vsip::scalar_f>(2.0));

  // Test dividing a matrix by a scalar.
  matrix_scalarf = -3.2;
  check_entry (vsip::div (matrix_scalarf, static_cast<vsip::scalar_f>(1.6)),
	       2, 1, static_cast<vsip::scalar_f>(-2.0));

  // Test nesting function calls.  This should yield a matrix of 0.0's.
  matrix_scalarf = 0.0;
  matrix_scalarf = vsip::acos (vsip::cos (matrix_scalarf));
  check_entry (matrix_scalarf, 1, 1, static_cast<vsip::scalar_f>(0.0));

  // Test arithmetic on matrixes.
  matrix_scalarf = 1.1;
  matrix_scalarf = static_cast<vsip::scalar_f>(2.0) * matrix_scalarf
    + matrix_scalarf;
  check_entry (matrix_scalarf, 1, 1, static_cast<vsip::scalar_f>(3.3));
  matrix_scalarf = 1.1;
  matrix_scalarf = static_cast<vsip::scalar_f>(2.0) * matrix_scalarf
    + (static_cast<vsip::scalar_f>(4.0) + matrix_scalarf);
  check_entry (matrix_scalarf, 1, 1, static_cast<vsip::scalar_f>(7.3));

  // Test incrementing a matrix.
  matrix_scalarf = 0.0;
  matrix_scalarf += 1.1;
  check_entry (matrix_scalarf, 1, 1, static_cast<vsip::scalar_f>(1.1));

  return EXIT_SUCCESS;
}

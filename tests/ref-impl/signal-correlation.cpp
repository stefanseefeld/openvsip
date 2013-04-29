/***********************************************************************

  File:   signal-correlation.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   07/21/2004

  Contents: Very simple tests of the correlation class.

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

/* Two-dimensional correlations are not supported by the reference
   VSIP Library so they are not implemented in this VSIPL++
   implementation and not tested.  */

/***********************************************************************
  Included Files
***********************************************************************/

#include <cstdlib>
#include "test.hpp"
#include <vsip/initfin.hpp>
#include <vsip/signal.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int argc, char** argv)
{
  using namespace vsip;
  vsipl		v(argc, argv);

  /* Begin testing of correlations.  */

  /* Correlating with 0.0 reference coefficients should yield answers
     with only 0.0.  */

  length_type const
		reference_size = 4;
			/* The reference size.  */

  length_type const
		input_size = 8;
			/* The input vector size.  */

  Correlation<const_Vector, support_min>
		zero_corr ((Domain<1>(reference_size)),
			   (Domain<1>(input_size)));
			/* A correlation object.  */

  Vector<>	zeroes (reference_size, 0.0);
			/* A vector containing only zeroes.  */

  /* Check the assessors.  */
  insist (equal (zero_corr.reference_size (), (Domain<1>(reference_size))));
  insist (equal (zero_corr.input_size (), (Domain<1>(input_size))));
  insist (equal (zero_corr.output_size (),
		 (Domain<1>(input_size - reference_size + 1))));
  insist (equal (zero_corr.support (), support_min));

  /* Test the operation.  */
  Vector<>	input (input_size);
			/* A vector with correlation input.  */

  Vector<>	answer (input_size - reference_size + 1);
			/* A vector to store the correlation answer.  */

  for (vsip::index_type i = 0; i < input_size; ++i)
    input.put (i, i);

  zero_corr (unbiased, zeroes, input, answer);
  for (vsip::index_type i = 0; i < zero_corr.output_size ().size (); ++i)
    insist (equal (answer.get (i), static_cast<scalar_f>(0.0)));

  /* Correlating with 1.0 reference coefficients should yield answers
     equalling sums.  */
  
  Vector<>	ones (reference_size, 1.0);
			/* A vector containing only ones.  */

  Correlation<const_Vector, support_min>
		one_corr ((Domain<1>(reference_size)),
			  (Domain<1>(input_size)));
			/* A correlation object.  */

  /* Check the assessors.  */
  insist (equal (one_corr.reference_size (), (Domain<1>(reference_size))));
  insist (equal (one_corr.input_size (), (Domain<1>(input_size))));
  insist (equal (one_corr.output_size (),
		 (Domain<1>(input_size - reference_size + 1))));
  insist (equal (one_corr.support (), support_min));

  /* Check the correlations.  */
  one_corr (unbiased, ones, input, answer);
  
  insist (equal
	  (answer.get (0),
	   (static_cast<scalar_f>(1.0) / reference_size) *
	   (input.get (0) + input.get (1) + input.get (2) + input.get (3))));
  insist (equal
	  (answer.get (1),
	   (static_cast<scalar_f>(1.0) / reference_size) *
	   (input.get (1) + input.get (2) + input.get (3) + input.get (4))));

  return EXIT_SUCCESS;

  /* Test copying and assignment.  */
  
  one_corr = zero_corr;
  one_corr (biased, zeroes, input, answer);
  for (vsip::index_type i = 0; i < one_corr.output_size ().size (); ++i)
    insist (equal (answer.get (i), static_cast<scalar_f>(0.0)));
  
  Correlation<const_Vector, support_min>
		zero_corr_2 (one_corr);
  zero_corr_2 (unbiased, zeroes, input, answer);
  for (vsip::index_type i = 0; i < zero_corr_2.output_size ().size (); ++i)
    insist (equal (answer.get (i), static_cast<scalar_f>(0.0)));

  /* End testing correlation.  */

  return EXIT_SUCCESS;
}

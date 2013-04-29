/***********************************************************************

  File:   signal-convolution.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   07/20/2004

  Contents: Very simple tests of the convolution class.

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

/* Two-dimensional convolutions are not supported by the reference
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

  /* Begin testing of convolutions.  */

  /* Convolving with 0.0 kernel coefficients should yield answers with
     only 0.0.  */

  length_type const
		kernel_size = 4;
			/* The kernel size, not the kernel order.  */

  length_type const
		input_size = 8;
			/* The input vector size.  */

  Vector<>	zeroes (kernel_size, 0.0);
			/* A vector containing only zeroes.  */

  Convolution<const_Vector, nonsym, support_min>
		zero_conv (zeroes, input_size
			   /*, decimation = 1 */);
			/* A convolution object with a filter
			   containing only zeroes.  */

  /* Check the assessors.  */
  insist (equal (zero_conv.kernel_size (), (Domain<1>(kernel_size))));
  insist (equal (zero_conv.filter_order (), zero_conv.kernel_size ()));
  insist (equal (zero_conv.symmetry (), nonsym));
  insist (equal (zero_conv.input_size (), (Domain<1>(input_size))));
  insist (equal (zero_conv.output_size (),
		 (Domain<1>(input_size - kernel_size + 1))));
  insist (equal (zero_conv.support (), support_min));
  insist (equal (zero_conv.decimation (), static_cast<length_type>(1)));

  /* Test the operation.  */
  Vector<>	input (input_size);
			/* A vector with convolution input.  */

  Vector<>	answer (input_size - kernel_size + 1);
			/* A vector to store the convolution answer.  */

  for (vsip::index_type i = 0; i < input_size; ++i)
    input.put (i, i);

  zero_conv (input, answer);
  for (vsip::index_type i = 0; i < zero_conv.output_size ().size (); ++i)
    insist (equal (answer.get (i), static_cast<scalar_f>(0.0)));

  /* Convolving with 1.0 kernel coefficients should yield answers 
     equalling sums.  */
  
  Vector<>	ones (kernel_size, 1.0);
			/* A vector containing only ones.  */

  Convolution<const_Vector, nonsym, support_min>
		one_conv (ones, input_size
			   /*, decimation = 1 */);
			/* A convolution object with a filter
			   containing only ones.  */

  /* Check the assessors.  */
  insist (equal (one_conv.kernel_size (), (Domain<1>(kernel_size))));
  insist (equal (one_conv.filter_order (), one_conv.kernel_size ()));
  insist (equal (one_conv.symmetry (), nonsym));
  insist (equal (one_conv.input_size (), (Domain<1>(input_size))));
  insist (equal (one_conv.output_size (),
		 (Domain<1>(input_size - kernel_size + 1))));
  insist (equal (one_conv.support (), support_min));
  insist (equal (one_conv.decimation (), static_cast<length_type>(1)));

  /* Check the convolutions.  */
  one_conv (input, answer);
  insist (equal
	  (answer.get (0),
	   input.get (0) + input.get (1) + input.get (2) + input.get (3)));
  insist (equal
	  (answer.get (1),
	   input.get (1) + input.get (2) + input.get (3) + input.get (4)));

  return EXIT_SUCCESS;

  /* Test copying and assignment.  */
  
  one_conv = zero_conv;
  one_conv (input, answer);
  for (vsip::index_type i = 0; i < one_conv.output_size ().size (); ++i)
    insist (equal (answer.get (i), static_cast<scalar_f>(0.0)));
  
  Convolution<const_Vector, nonsym, support_min>
		zero_conv_2 (one_conv);
  zero_conv_2 (input, answer);
  for (vsip::index_type i = 0; i < zero_conv_2.output_size ().size (); ++i)
    insist (equal (answer.get (i), static_cast<scalar_f>(0.0)));

  /* End testing convolution.  */

  return EXIT_SUCCESS;
}

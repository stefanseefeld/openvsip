/***********************************************************************

  File:   signal-fft.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   07/13/2002

  Contents: Very simple tests of the Fft class.

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

#include <stdlib.h>
#include <utility>
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
  vsip::vsipl	v(argc, argv);

  /* Begin testing of FFTs.  Check only for compilation and execution,
     not actual values.  */

  vsip::length_type const
		input_length = 32;
  vsip::Vector<vsip::cscalar_f>
		vector_cscalar_f (input_length, vsip::cscalar_f ());
  vsip::Vector<vsip::cscalar_f>
		vector_cscalar_answer (input_length);

  // Test return-by-value object.
  vsip::Fft<vsip::const_Vector, vsip::cscalar_f, vsip::cscalar_f, vsip::fft_fwd>
		fft_ccfv ((vsip::Domain<1>(input_length)),
			  static_cast<vsip::scalar_f>(2.0));
  /* Use default parameters:
     0, vsip::SINGLE, vsip::by_value, 0, vsip::alg_time  */

  /* Test some accessors.  */

  insist (equal (fft_ccfv.input_size (), (vsip::Domain<1>(input_length))));
  insist (equal (fft_ccfv.output_size (), fft_ccfv.input_size ()));
  insist (equal (fft_ccfv.scale (), static_cast<vsip::scalar_f>(2.0)));
  insist (equal (fft_ccfv.forward (), true));

  /* Test the operator.  */

  vector_cscalar_answer = fft_ccfv (vector_cscalar_f);

  // Test return-by-reference object.
  
  vsip::Fft<vsip::const_Vector, vsip::cscalar_f, vsip::cscalar_f,
            vsip::fft_fwd, vsip::by_reference>
		fft_ccfr ((vsip::Domain<1>(input_length)),
			  static_cast<vsip::scalar_f>(2.0));
  /* Use default parameters:
     0, vsip::alg_time  */

  /* Test some accessors.  */

  insist (equal (fft_ccfr.input_size (), (vsip::Domain<1>(input_length))));
  insist (equal (fft_ccfr.output_size (), fft_ccfr.input_size ()));
  insist (equal (fft_ccfr.scale (), static_cast<vsip::scalar_f>(2.0)));
  insist (equal (fft_ccfr.forward (), true));

  /* Test the operator.  */

  fft_ccfr (vector_cscalar_f, vector_cscalar_answer);

  // Test multiple-Fft objects.

  vsip::Matrix<vsip::cscalar_f>
		matrix_cscalar_f (input_length, input_length,
				  vsip::cscalar_f ());
  vsip::Matrix<vsip::cscalar_f>
		matrix_cscalar_answer (input_length, input_length);

  vsip::Fftm<vsip::cscalar_f, vsip::cscalar_f, 0, vsip::fft_fwd,
             vsip::by_value>
		fft_mccfv ((vsip::Domain<2>(input_length, input_length)),
			   static_cast<vsip::scalar_f>(2.0));
  /* Use default parameters:
     0, vsip::alg_time  */

  /* Test some accessors.  */

  insist (equal (fft_mccfv.input_size (),
		 vsip::Domain<2>(input_length, input_length)));
  insist (equal (fft_mccfv.output_size (), fft_mccfv.input_size ()));
  insist (equal (fft_mccfv.scale (), static_cast<vsip::scalar_f>(2.0)));
  insist (equal (fft_mccfv.forward (), true));

  /* Test the operator.  */

  matrix_cscalar_answer = fft_mccfv (matrix_cscalar_f);

  vsip::Fftm<vsip::cscalar_f, vsip::cscalar_f, 0, vsip::fft_fwd,
             vsip::by_reference>
		fft_mccfr ((vsip::Domain<2>(input_length, input_length)),
			   static_cast<vsip::scalar_f>(3.0));
  /* Use default parameters:
     0, vsip::alg_time  */

  /* Test some accessors.  */

  insist (equal (fft_mccfr.input_size (),
		 vsip::Domain<2>(input_length, input_length)));
  insist (equal (fft_mccfr.output_size (), fft_mccfr.input_size ()));
  insist (equal (fft_mccfr.scale (), static_cast<vsip::scalar_f>(3.0)));
  insist (equal (fft_mccfr.forward (), true));

  /* Test the operator.  */

  fft_mccfr (matrix_cscalar_f, matrix_cscalar_answer);

  // Create a real->complex Fft object.
  vsip::Fft<vsip::const_Vector, vsip::scalar_f, vsip::cscalar_f, 0,
            vsip::by_reference>
		fft_vrcfr (input_length, static_cast<vsip::scalar_f>(3.0));


  return EXIT_SUCCESS;
}

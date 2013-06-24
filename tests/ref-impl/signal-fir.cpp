/***********************************************************************

  File:   signal-fir.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   08/14/2002

  Contents: Very simple tests of the Fir filter class.

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
#include <vsip/signal.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>

/***********************************************************************
  Function Declarations
***********************************************************************/

template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
inline bool
checkVector (const vsip::length_type            out_length,
	     const vsip::Vector<T0, Block0>& answer,
	     const vsip::Vector<T1, Block1>& vec) VSIP_NOTHROW;

/***********************************************************************
  Inline Function Definitions
***********************************************************************/

template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
inline bool
checkVector (const vsip::length_type            out_length,
	     const vsip::Vector<T0, Block0>& answer,
	     const vsip::Vector<T1, Block1>& vec) VSIP_NOTHROW
{
  if (out_length != answer.size ())
    return false;

  for (vsip::index_type idx = 0; idx < vec.size (); ++idx)
    if (!equal(answer.get (idx), vec.get (idx)))
      return false;

  return true;
}

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int argc, char** argv)
{
  vsip::vsipl	v(argc, argv);

  /* Begin testing of Fir.  */

  /* Fir filtering with 0.0 kernel coefficients should yield answers
     with only 0.0.  */

  vsip::length_type
		input_length = 3;
  vsip::Fir<> 	fir0 (vsip::Vector<>(2, 0.0), input_length);
  vsip::Vector<>
		input0 (input_length, 0.0);
  insist (equal (fir0.output_size (), static_cast<vsip::length_type>(3)));
  vsip::Vector<>
		output0 (3, 0.0);
  vsip::Vector<>
		answer0 (3, 0.0);
  vsip::length_type
		out_length;
  out_length = fir0 (input0, output0);
  insist (checkVector (out_length, output0, answer0));
  out_length = fir0 (input0, output0);
  insist (checkVector (out_length, output0, answer0));
  out_length = fir0 (input0, output0);
  insist (checkVector (out_length, output0, answer0));

  input0 = 1.0; output0 = 0.0;
  out_length = fir0 (input0, output0);
  insist (checkVector (out_length, output0, answer0));
  out_length = fir0 (input0, output0);
  insist (checkVector (out_length, output0, answer0));
  out_length = fir0 (input0, output0);
  insist (checkVector (out_length, output0, answer0));

  /* Fir filter with 1.0 kernel coefficients.  */

  vsip::Fir<> 	fir1 (vsip::Vector<>(2, 1.0), input_length);
  input0 = 0.0; output0 = 0.0;
  out_length = fir1 (input0, output0);
  insist (checkVector (out_length, output0, answer0));
  out_length = fir1 (input0, output0);
  insist (checkVector (out_length, output0, answer0));
  out_length = fir1 (input0, output0);
  insist (checkVector (out_length, output0, answer0));

  input0 = 1.0; output0 = 0.0;
  vsip::Vector<>
		answer1 (3, 2.0);
  answer1.put (0, 1.0);
  vsip::Vector<>
		answer2 (3, 2.0);
  out_length = fir1 (input0, output0);
  insist (checkVector (out_length, output0, answer1));
  out_length = fir1 (input0, output0);
  insist (checkVector (out_length, output0, answer2));
  out_length = fir1 (input0, output0);
  insist (checkVector (out_length, output0, answer2));


  /* Test resetting.  */

  fir1.reset ();
  out_length = fir1 (input0, output0);
  insist (checkVector (out_length, output0, answer1));

  /* Test assignment operator and copy constructor.  */

#if VSIP_HAS_EXCEPTIONS
  try
  {
#endif
    vsip::Fir<> 	fir2 (fir1);
    out_length = fir2 (input0, output0);
    insist (checkVector (out_length, output0, answer2));
    fir2 = fir0;
    out_length = fir2 (input0, output0);
    insist (checkVector (out_length, output0, answer0));
#if VSIP_HAS_EXCEPTIONS
  }
  // C-VSIPL doesn't provide state-preserving assignment.
  catch (ovxx::unimplemented const &) {}
#endif
  /* Test decimations equaling 1 and 2.  */

  input_length = 5;
  vsip::Fir<> 	fir3 (vsip::Vector<>(3, 1.0), input_length);
  vsip::Vector<>
		input1 (input_length);
  insist (equal (fir3.output_size (), input_length));
  vsip::Vector<>
		output1 (input_length, 0.0);
  vsip::Vector<>
    		answer_d1_1 (input_length),
		answer_d1_2 (input_length);
  for (vsip::index_type idx = 0; idx < input_length; ++idx) {
    input1.put (idx, idx + 1);
    answer_d1_1.put (idx, 3 * idx);
    answer_d1_2.put (idx, 3 * idx);
  }
  answer_d1_1.put (0, 1);
  answer_d1_1.put (1, 3);
  answer_d1_2.put (0, 10);
  answer_d1_2.put (1, 8);

  out_length = fir3 (input1, output1);
  insist (checkVector (out_length, output1, answer_d1_1));
  out_length = fir3 (input1, output1);
  insist (checkVector (out_length, output1, answer_d1_2));

  vsip::Fir<vsip::VSIP_DEFAULT_VALUE_TYPE, vsip::nonsym>
		fir4 (vsip::Vector<> (3, 1.0), input_length, 2);
  vsip::Vector<>
		output1b (fir4.output_size (), 0.0);
  vsip::Vector<>
    		answer_d2_1 ((input_length + 1) / 2),
		answer_d2_2 (input_length / 2);
  answer_d2_1.put (0, answer_d1_1.get (0));
  answer_d2_1.put (1, answer_d1_1.get (2));
  answer_d2_1.put (2, answer_d1_1.get (4));
  answer_d2_2.put (0, answer_d1_2.get (1));
  answer_d2_2.put (1, answer_d1_2.get (3));
  out_length = fir4 (input1, output1b);
  insist (checkVector (out_length, output1b, answer_d2_1));

  /* End testing of Fir.  */

  return EXIT_SUCCESS;
}

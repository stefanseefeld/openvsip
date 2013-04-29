/***********************************************************************

  File:   signal-histogram.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   08/03/2004

  Contents: Very simple tests of the Histogram class.

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
#include <vsip/signal.hpp>
#include <vsip/vector.hpp>

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int argc, char** argv)
{
  using namespace vsip;

  vsipl		v(argc, argv);

  const length_type
		len = 25;
			/* The length of a created vector.  */

  scalar_f const
		min_value = 2.0;
			/* The smallest value for the histogram.  */

  scalar_f const
		max_value = 20.0;
			/* All values greater than or equal to this
			   value are too large for the histogram
			   bins.  */

  Vector<>	data (len);

  for (vsip::index_type idx = len; idx-- > 0; )
    data.put (idx, static_cast<scalar_f>(idx));

  Histogram<>	histo (min_value, max_value, /* number of bins = */ 4);

  Vector<>	answer (histo (data));
  insist (equal (answer.get (0), static_cast<scalar_f>(2)));
  insist (equal (answer.get (1), static_cast<scalar_f>(9)));
  insist (equal (answer.get (2), static_cast<scalar_f>(9)));
  insist (equal (answer.get (3), static_cast<scalar_f>(5)));
  
  answer = histo (data, /* accumulate= */ true);
  insist (equal (answer.get (0), static_cast<scalar_f>(2 * 2)));
  insist (equal (answer.get (1), static_cast<scalar_f>(2 * 9)));
  insist (equal (answer.get (2), static_cast<scalar_f>(2 * 9)));
  insist (equal (answer.get (3), static_cast<scalar_f>(2 * 5)));

  return EXIT_SUCCESS;
}

/***********************************************************************

  File:   signal.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   07/29/2004

  Contents: Very simple tests of signal functions.

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

  vsipl	v(argc, argv);

#if 0 /* tvcpp0p8 VSIPL does not implement freqswap.  */
  /* Test freqswap.  */

  Vector<>	vec (3, 1.0);
  vec.put (0, 2.0);
  vec.put (1, 3.0);
  Vector<>	answer (freqswap (vec));
  insist (equal (answer.get (0), static_cast<scalar_f>(1.0)));
  insist (equal (answer.get (1), static_cast<scalar_f>(2.0)));
  insist (equal (answer.get (2), static_cast<scalar_f>(3.0)));
#endif /* tvcpp0p8 VSIPL does not implement freqswap.  */

  return EXIT_SUCCESS;
}

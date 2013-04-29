/***********************************************************************

  File:   random.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   07/26/2004

  Contents: Very simple test of random number generation.

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
#include <vsip/initfin.hpp>
#include <vsip/random.hpp>
#include <vsip/support.hpp>

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int argc, char** argv)
{
  using namespace vsip;

  vsipl		v(argc, argv);

  /* Create some random number generators.  */

  vsip::Rand<>
		rand_scalar_true (0, /* portable= */ true);

  vsip::Rand<scalar_f>
		rand_scalar_false (0, /* portable= */ false);

  vsip::Rand<cscalar_f>
		rand_cscalar_true (0, /* portable= */ true);

  vsip::Rand<cscalar_f>
		rand_cscalar_false (0, 23, 3, /* portable= */ false);

  /* Obtain some randomly generated numbers.  We have no easy way to
     ensure they are nearly random.  */

  rand_scalar_true.randu ();
  rand_scalar_false.randn ();
  rand_cscalar_true.randu (17);
  rand_cscalar_false.randn (17);

  return EXIT_SUCCESS;
}

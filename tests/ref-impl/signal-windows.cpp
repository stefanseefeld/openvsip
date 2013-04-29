/***********************************************************************

  File:   signal-windows.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   07/21/2003

  Contents: Very simple tests of the signal window functions.

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

float
blackman_ref(
   int		idx,
   int		len)
{
  const vsip::scalar_f
		pi = 4.0 * vsip::atan (static_cast<vsip::scalar_f>(1.0));
			/* One of the five most important mathematical
			   constants.  */

  return (0.42f
	   - 0.5f  * vsip::cos(2.0f * pi * idx / (len - 1))
	   + 0.08f * vsip::cos(4.0f * pi * idx / (len - 1)));
}



//
// Note:
//  - vec.length() is the length of the input vector, which may be
//    a subview of a blackman weight vector.  Subviews must start
//    at 0.
//  - len is the length of the original blackman weight vector.
template <typename T,
	  typename Block>
void
check_blackman(
   vsip::const_Vector<T, Block>	vec,
   int				offset,
   int				len)
{
   for (vsip::index_type i=0; i<vec.length(); ++i)
      check_entry (vec, i, blackman_ref(i+offset, len));
}


void
test_cheby(vsip::length_type len, vsip::scalar_f ripple)
{
  using vsip::Vector;
  using vsip::scalar_f;
  using vsip::index_type;
  using vsip::cheby;

  Vector<scalar_f> vec1 = cheby(len, ripple);
  Vector<scalar_f> vec2(len);

  vec2 = cheby(len, ripple);

  for (index_type i=0; i<len; ++i)
    assert(equal(vec1.get(i), vec2.get(i)));
}



int
main (int argc, char** argv)
{
  vsip::vsipl	v(argc, argv);

  const vsip::length_type
		len = 32;
			/* The length of a created vector.  */

  const vsip::scalar_f
		pi = 4.0 * vsip::atan (static_cast<vsip::scalar_f>(1.0));
			/* One of the five most important mathematical
			   constants.  */

  vsip::index_type	idx;	/* Index to check.  */

  /* Test blackman.  */
  check_blackman(vsip::blackman(len), 0, len);

  vsip::Vector<vsip::scalar_f> vec = vsip::blackman(len);
  check_blackman(vec, 0, len);

  check_blackman(vec.get(vsip::Domain<1>(0,1,len-1)), 0, len); // note #1
  check_blackman(vec.get(vsip::Domain<1>(1,1,len-2)), 1, len);

  // Note:
  //  1. The reference implementation of blackman was returning a
  //     vector with the view_ field set to the value of
  //     vsip_vcreate_blackman_f(), but was not updating the block_
  //     to be consistent.  Since the vector's view_ and block_ are
  //     unrelated, subviews did not work.

  idx = 2;
  check_entry (vsip::blackman (len), idx, blackman_ref(idx, len));

  idx = 17;
  check_entry (vsip::blackman (len), idx, blackman_ref(idx, len));

  /* Test hanning.  */

  idx = 2;
  check_entry (vsip::hanning (len), idx,
	       static_cast<vsip::scalar_f>
	       (0.5
		* (1.0 - vsip::cos
		   (static_cast<vsip::scalar_f>((2.0 * pi * (idx + 1))
						/ (len + 1))))));

  idx = 17;
  check_entry (vsip::hanning (len), idx,
	       static_cast<vsip::scalar_f>
	       (0.5
		* (1.0 - vsip::cos
		   (static_cast<vsip::scalar_f>((2.0 * pi * (idx + 1))
						/ (len + 1))))));

  /* Test cheby.  */
  test_cheby(16,  1.f);
  test_cheby(17,  1.f);
  test_cheby(18,  1.f);
  test_cheby(256, 1.f);
  test_cheby(257, 1.f);
  test_cheby(258, 1.f);

  return EXIT_SUCCESS;
}

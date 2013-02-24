/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   parallel iterator function

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/selgen.hpp>
#include <vsip_csl/pi.hpp>
#include <vsip_csl/output.hpp>
#include <vsip_csl/profile.hpp>

using namespace vsip;
using namespace vsip_csl;

// This function "reverts" iteration, by
// mapping `i` to `N - i`
index_type revert(index_type i) { return 7 - i;}

// Generate an iterator expression
template <typename I>
typename enable_if<pi::is_iterator<I>, pi::Map>::type
revert(I i) { return pi::Map(revert, i);}

int main(int argc, char **argv)
{
  vsipl init(argc, argv);

  int const N = 8;

  Vector<> input = ramp(7., -1., N);
  Vector<> output(N);

  pi::Iterator<> i;
  // Assign from input to output, but reverse the order.
  output(i) = input(revert(i));
  std::cout << "input : \n" << input << std::endl;
  std::cout << "output : \n" << output << std::endl;
}

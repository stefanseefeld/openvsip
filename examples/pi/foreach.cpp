/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   sumsqval using pi::foreach

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/selgen.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/pi.hpp>
#include <vsip_csl/output.hpp>

using namespace vsip;
using namespace vsip_csl;

namespace example
{
// Compute the sum of the magnitude squared of an input view.
struct Summagsqval
{
  typedef float result_type;

  template <typename B>
  result_type operator()(Vector<complex<float>, B> v) const
  { return sumval(magsq(v));}
};
}

int main(int argc, char **argv)
{
  vsipl init(argc, argv);

  int const N = 8;

  Matrix<complex<float> > input(N, N);
  for (index_type r = 0; r != input.size(0); ++r)
    input.row(r) = ramp(static_cast<float>(N*r), 1.f, N);

  Vector<float> output(N);

  pi::Iterator<> i;
  output(i) = pi::foreach<example::Summagsqval>(input.row(i));

  std::cout << "input : \n" << input << std::endl;
  std::cout << "output : \n" << output << std::endl;
}

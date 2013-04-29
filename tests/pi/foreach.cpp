/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/selgen.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/pi.hpp>
#include <vsip_csl/test.hpp>

using namespace vsip;

// Compute the sum of the magnitude squared of an input view.
struct Summagsqval
{
  typedef float result_type;

  template <typename B>
  result_type operator()(Vector<complex<float>, B> v) const
  { return sumval(magsq(v));}
};

int main(int argc, char **argv)
{
  vsipl init(argc, argv);

  int const N = 1024;

  Matrix<complex<float> > input(N, N);
  for (index_type r = 0; r != input.size(0); ++r)
    input.row(r) = ramp(static_cast<float>(N*r), 1.f, N);

  Vector<float> reference(N);

  Summagsqval op;
  for (index_type i = 0; i != reference.size(); ++i)
    reference(i) = op(input.row(i));

  Vector<float> output(N);

  vsip_csl::pi::Iterator<> i;
  output(i) = vsip_csl::pi::foreach<Summagsqval>(input.row(i));
  test_assert(vsip_csl::view_equal(output, reference));
}

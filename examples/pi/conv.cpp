/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   multiple convolution using pi::foreach

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/selgen.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/strided.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip_csl/pi.hpp>
#include <vsip_csl/output.hpp>
#include <vsip_csl/test.hpp>
#include <iostream>

using namespace vsip;
namespace p = vsip_csl::profile;
namespace pi = vsip_csl::pi;

int main(int argc, char **argv)
{
  vsipl init(argc, argv);

  length_type in_len = 65536; // input size
  length_type krn_len = 4; // kernel size
  length_type dec = 1; // decimation
  length_type out_len = (in_len - 1) / dec - (krn_len - 1) / dec + 1;

  Vector<float> kernel(krn_len);
  kernel(0) = 0.1;
  kernel(1) = 0.4;
  kernel(2) = 0.4;
  kernel(3) = 0.1;

  Matrix<float> input(32, in_len);
  for (index_type r = 0; r != input.size(0); ++r)
    input.row(r) = ramp(static_cast<float>(in_len*r), 1.f, in_len);


  // Make sure the output block has well-aligned rows, despite 'odd' sizes.
  typedef Layout<2, tuple<>, aligned_128, interleaved_complex> layout_type;
  Matrix<float, vsip_csl::Strided<2, float, layout_type> > reference(32, out_len);
  Matrix<float, vsip_csl::Strided<2, float, layout_type> > output(32, out_len);

  Convolution<const_Vector, nonsym, support_min, float> conv(kernel, in_len, dec);

  p::Timer t;
  t.start();
  for (index_type i = 0; i != input.size(0); ++i)
    conv(input.row(i), reference.row(i));
  t.stop();
  std::cout << "serial version : " << t.delta() << " seconds" << std::endl;

  t.start();
  pi::Iterator<> i;
  output.row(i) = pi::foreach(conv, input.row(i));
  t.stop();
  std::cout << "parallel iterator version : " << t.delta() << " seconds" << std::endl;
  test_assert(vsip_csl::view_equal(output, reference));
}

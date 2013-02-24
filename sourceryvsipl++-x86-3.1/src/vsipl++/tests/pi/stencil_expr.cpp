/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   Stencil expression tests.

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip_csl/pi.hpp>
#include <vsip_csl/error_db.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

using namespace vsip;
using namespace vsip_csl;

void test_shift(Domain<2> const &dom, int shift_y, int shift_x)
{
  Matrix<float> input(dom[0].length(), dom[1].length());
  for (index_type x = 0; x != dom[1].length(); ++x)
    for (index_type y = 0; y != dom[0].length(); ++y)
      input(y, x) = x + 10.f*y;

  Matrix<float> output(dom[0].length(), dom[1].length());

  pi::Iterator<> i;
  pi::Iterator<> j;

  output(i, j) = input(i + shift_y, j + shift_x);

  // Define appropriate subdomains for comparison.
  Domain<2> out_dom(dom[0].length() - vsip::mag(shift_y),
                    dom[1].length() - vsip::mag(shift_x));
  Domain<2> in_dom = out_dom;
  if (shift_y > 0) in_dom.impl_at(0) = in_dom[0] + shift_y;
  else out_dom.impl_at(0) = out_dom[0] - shift_y;
  if (shift_x > 0) in_dom.impl_at(1) = in_dom[1] + shift_x;
  else out_dom.impl_at(1) = out_dom[1] - shift_x;

  double error = error_db(input(in_dom), output(out_dom));
  if (error >= -100)
  {
    std::cout << "input" << input << std::endl;
    std::cout << "output" << output << std::endl;
  }
  test_assert(error < -100);
}

void test_expr(Domain<2> const &dom)
{
  Matrix<float> input(dom[0].length(), dom[1].length());
  for (index_type x = 0; x != dom[1].length(); ++x)
    for (index_type y = 0; y != dom[0].length(); ++y)
      input(y, x) = x + 10.f*y;

  Matrix<float> output(dom[0].length(), dom[1].length());

  pi::Iterator<> i;
  pi::Iterator<> j;

  // The generated kernel should be an identity matrix. We simply test
  // expression evaluation here.
  output(i, j) = (input(i, j) + input(i + 1, j) - 1 * input(i + 1, j) +
                  input(i, j - 1) * 1 - input(i, j - 1) / 1 +
                  (input(i - 1, j) - input(i - 1, j))/2 +
                  2*(input(i, j + 1) - input(i, j + 1)));

  double error = error_db(input, output);
  if (error >= -100)
  {
    std::cout << "input" << input << std::endl;
    std::cout << "output" << output << std::endl;
  }
  test_assert(error < -100);
}



int main(int argc, char **argv)
{
  vsipl init(argc, argv);
  test_shift(Domain<2>(256, 256), 0, 0);
  test_shift(Domain<2>(256, 256), 5, 0);
  test_shift(Domain<2>(256, 256), 0, 5);
  test_shift(Domain<2>(256, 256), -5, 0);
  test_shift(Domain<2>(256, 256), 0, -5);

  test_expr(Domain<2>(256, 256));

  return 0;
}

/* Copyright (c) 2010, 2011 CodeSourcery, Inc.  All rights reserved. */

/* Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
         disclaimer in the documentation and/or other materials
         provided with the distribution.
       * Neither the name of CodeSourcery nor the names of its
         contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.

   THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */

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
  typedef Layout<2, tuple<>, aligned_128, array> layout_type;
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

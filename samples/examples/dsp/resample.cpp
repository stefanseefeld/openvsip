/* Copyright (c) 2011 CodeSourcery, Inc.  All rights reserved. */

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
///   Change the sample rate of a signal by a ratio of integers three
///   ways: 1) using a standard FIR filter, 2) using a decimating
///   FIR filter and 3) using a more computationally efficient 
///   polyphase FIR filter.

#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/selgen.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/test.hpp>

#include "fir_window.hpp"

int
main(int argc, char **argv)
{
  using namespace vsip;
  
  vsipl init(argc, argv);

  // Change sample rate by a ratio of two integers, P and Q
  length_type P = 5;               // upsample rate, number of filter banks
  length_type Q = 3;               // downsample rate
  length_type M = 64 * 1024;       // input signal length
  length_type S = 8;               // subfilter length
  length_type L = P * S;           // filter length
  length_type NN = M * P;          // tmp view size
  length_type N = ceil(static_cast<scalar_f>(NN) / Q);  // output length

  // Construct input signal
  Vector<scalar_f> x(M, scalar_f());
  Vector<int> i = ramp<int>(0, 1, M);
  x = sin(10 * 2 * M_PI * (i+1) / M);

  // Calculate prototype filter coefficients
  // 
  // Since P > Q, a low-pass filter with a cutoff frequency of pi/P is
  // sufficient as an anti-aliasing (interpolation) filter as well as to 
  // remove spectral images resulting from the downsampling step.
  test_assert(P > Q);
  Vector<scalar_f> h = fir_window(M_PI / P, blackman(L));
  h *= P;  // scale coefficients for zero gain


  /////////////////////////////////////////////////////////////////
  // Simple resampling
  // 
  //   Upsample, filter and downsample.
  /////////////////////////////////////////////////////////////////

  Vector<scalar_f> xx(NN, scalar_f());
  Vector<scalar_f> yy(NN, scalar_f());
  Vector<scalar_f> y1(N, scalar_f());
  Fir<scalar_f> lpf(h, NN);
  Domain<1> up(0, P, M);      // selects values spaced P samples apart
  Domain<1> down(0, Q, N);    // selects values spaced Q samples apart

  // Upsample
  xx(up) = x;

  // Filter
  lpf(xx, yy);

  // Downsample
  y1 = yy(down);


  /////////////////////////////////////////////////////////////////
  // Simple resampling using FIR filter decimation
  // 
  //   Upsample, then filter and downsample in one step.
  //   Best performance when Q is large relative to P.
  /////////////////////////////////////////////////////////////////

  Vector<scalar_f> y2(N, scalar_f());
  Fir<scalar_f> lpf2(h, NN, Q);

  // Upsample
  xx(up) = x;

  // Filter and downsample
  lpf2(xx, y2);


  /////////////////////////////////////////////////////////////////
  // Polyphase filtering
  //
  //   Combines the upsampling and filtering steps rather than
  //     the filtering and downsampling steps as above.
  //   Reduced computational requirements (avoids multiplying 
  //     coefficients with zero-stuffed inputs).
  //   Best performance when P is large relative to Q.
  /////////////////////////////////////////////////////////////////

  // Reorganize filter into P subfilters of length Q
  Matrix<scalar_f> hs(P, S);
  for (length_type r = 0; r < P; ++r)
    hs.row(r) = h(Domain<1>(r, P, S));
  Fir<scalar_f> lp1(hs.row(0), M);
  Fir<scalar_f> lp2(hs.row(1), M);
  Fir<scalar_f> lp3(hs.row(2), M);
  Fir<scalar_f> lp4(hs.row(3), M);
  Fir<scalar_f> lp5(hs.row(4), M);

  // Reorganize output into subvectors, each computed by
  // filtering the input with one of the subfilters.  This 
  // upsamples and filters at the same time.
  lp1(x, yy(Domain<1>(0, P, NN/P)));
  lp2(x, yy(Domain<1>(1, P, NN/P)));
  lp3(x, yy(Domain<1>(2, P, NN/P)));
  lp4(x, yy(Domain<1>(3, P, NN/P)));
  lp5(x, yy(Domain<1>(4, P, NN/P)));
  
  // Downsample
  Vector<scalar_f> y3(N, scalar_f());
  y3 = yy(down);


  // Verify results
  test_assert(vsip_csl::view_equal(y1, y2));
  test_assert(vsip_csl::view_equal(y1, y3));

  std::cout << "Done." << std::endl 
            << "All three methods produced identical results." << std::endl;
}

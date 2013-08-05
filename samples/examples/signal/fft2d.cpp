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
///   2-Dimensional FFT example.
///   Separate a two-dimensional dataset into spectral components and
///   find the strongest of two mixed-frequency signals.
///
/// Keywords: Fft, Index, mag, Matrix, maxval

#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/test.hpp>

using namespace vsip;


///////////////////////////////////////////////////////////////////////

int
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  
  // Create a 2-D complex->complex forward transform
  vsip::length_type N = 16;
  Fft<Matrix, cscalar_f, cscalar_f, fft_fwd, by_value>
    fft2d(Domain<2>(N, N), 1.0);

  // Allocate input and output buffers
  Matrix<cscalar_f> in(N, N, cscalar_f());
  Matrix<cscalar_f> out(N, N);

  // Fill input view with periodic signals at two frequencies, A and B,
  // making signal B the stronger of the two.  Each signal is composed
  // of components in the X and Y directions (along the rows and columns
  // respectively).
  float ma[2] = {2.0f, 5.0f};    // magnitude A
  float fa[2] = {2.0f, 3.0f};    // frequency A
  float mb[2] = {3.0f, 10.0f};   // magnitude B  <-- |B| > |A|
  float fb[2] = {11.0f, 13.0f};  // frequency B
  for (length_type i = 0; i < N; ++i)
    for (length_type j = 0; j < N; ++j)
    {
      float const w[2] = {2 * M_PI * i / N,  2 * M_PI * j / N};
      in(i, j) = polar(ma[0], fa[0] * w[0]) * polar(ma[1], fa[1] * w[1])
               + polar(mb[0], fb[0] * w[0]) * polar(mb[1], fb[1] * w[1]);
    }


  // Compute 2d FFT
  out = fft2d(in);


  // Search in frequency space for the point with the most energy.
  // This location gives each frequency component, which should 
  // correspond exactly to one of the two points above (as these
  // are the only frequencies involved in this simplified case).
  Index<2> idx;
  scalar_f maxv = maxval(mag(out), idx);
  (void)maxv;

  // Point B is found because the combined intensities of the 
  // frequencies are greater than those of A (|fb| > |fa|).
  test_assert(vsip_csl::equal(fb[0], static_cast<float>(idx[0])));
  test_assert(vsip_csl::equal(fb[1], static_cast<float>(idx[1])));
}

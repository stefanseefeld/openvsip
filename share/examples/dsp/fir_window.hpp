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
///   Calculate the coefficients for a low-pass FIR filter

#ifndef FIR_WINDOW_HPP
#define FIR_WINDOW_HPP

#include <vsip/vector.hpp>
#include <vsip/selgen.hpp>

#include "sinc.hpp"


// Computes the coefficients for a windowed FIR filter based on
// a given cutoff frequency.
// 
//    wc  The cutoff frequency in radians
//
//    w   The desired window coefficients (Blackman, Hanning or other)
//
// Defined as
//
//           wc
//   h(n) = ---- sinc[ wc * (n - M) ] w(n)
//           pi
//
//   for 0 <= n <= N-1, where N is the length of the input w
//
// Returns a vector of coefficients for a low-pass filter
//
vsip::Vector<vsip::scalar_f>
fir_window(
  vsip::scalar_f wc,
  vsip::const_Vector<vsip::scalar_f> w)
{
  using namespace vsip;

  length_type N = w.size();
  scalar_f M = (N - 1) / 2.f;
  Vector<scalar_f> n = ramp<scalar_f>(0, 1, N);
  Vector<scalar_f> wci = wc * (n - M);

  Vector<scalar_f> h = wc * sinc(wci) * w / M_PI;
  return h;
}


#endif // FIR_WINDOW_HPP

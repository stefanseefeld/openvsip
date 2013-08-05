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
///   Calculate the frequency response for a rational filter

#ifndef FREQZ_HPP
#define FREQZ_HPP

#include <vsip/vector.hpp>
#include <vsip/signal.hpp>


// Computes the frequency response of rational IIR filter
//
//    b    Feed-forward coefficients
//
//    a    Feed-back coefficients
//
//    N    Number of points to evaluate over the interval 0 --> pi
//
// Defined as
//
//             B(e^iw)
//   H(e^iw) = -------
//             A(e^iw)
//
//   where w is the angular frequency, 0 <= w <= pi
//
//   and x(e^iw) <--DTFT--> X(e^iw)
//
// Returns a complex vector containing the frequency response.
//
// Note: FIR filters can be designed with this equation by setting
// the denominator to a single value of 1.
//
vsip::Vector<vsip::cscalar_f>
freqz(
  vsip::const_Vector<vsip::scalar_f> b,
  vsip::const_Vector<vsip::scalar_f> a,
  vsip::length_type N)
{
  using namespace vsip;
  assert((N >= b.size()) && (N >= a.size()));

  typedef Fft<const_Vector, cscalar_f, cscalar_f, fft_fwd> fft_type;
  fft_type fft(Domain<1>(N), 1.0);

  typedef Vector<cscalar_f> vector_type;
  vector_type b_padded(N, cscalar_f());
  vector_type a_padded(N, cscalar_f());
  vector_type H(N);

  b_padded(Domain<1>(b.size())).real() = b;
  a_padded(Domain<1>(a.size())).real() = a;
    
  H = fft(b_padded) / fft(a_padded);
  return H;
}


// This overload handles the case where the denominator in the
// ratio of polynomial coefficients is a constant, as is the case
// with FIR filters (all the feed-back coefficients save the
// first are zero).
vsip::Vector<vsip::cscalar_f>
freqz(
  vsip::const_Vector<vsip::scalar_f> b,
  vsip::scalar_f sa,
  vsip::length_type N)
{
  vsip::const_Vector<vsip::scalar_f> a(1, sa);
  return freqz(b, a, N);
}


#endif // FREQZ_HPP

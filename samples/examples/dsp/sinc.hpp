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
///   Compute discrete points of the sinc function

#ifndef SINC_HPP
#define SINC_HPP

#include <vsip/vector.hpp>
#include <vsip/math.hpp>


// Sinc function for vectors
//
//    x   input vector of evaluation points
//
// Defined as
//                 sin(x)
//   y = sinc(x) = ------
//                  (x)
//
//   except at x==0, where sinc(0) = 1
//
// Returns a vector of computed sinc values
//
vsip::Vector<vsip::scalar_f>
sinc(vsip::const_Vector<vsip::scalar_f> x)
{
  using namespace vsip;

  length_type N = x.size();
  Vector<scalar_f> y(N);

  if (anytrue(x == 0.f))
  {
    // Avoid divide-by-zero error
    for (int i = 0; i < N; ++i)
      if (x(i) == 0.f)
        y(i) = scalar_f(1);
      else
        y(i) = sin(x(i)) / x(i);
  }
  else
  {
    y = sin(x) / x;
  }
  return y;
}


#endif // SINC_HPP

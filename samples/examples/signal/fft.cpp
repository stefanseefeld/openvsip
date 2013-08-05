/* Copyright (c) 2005, 2006, 2011 CodeSourcery, Inc.  All rights reserved. */

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
///   Simple FFT example

#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/error_db.hpp>

int
main(int argc, char **argv)
{
  using namespace vsip;
  
  vsipl init(argc, argv);
  
  typedef Fft<const_Vector, cscalar_f, cscalar_f, fft_fwd> f_fft_type;
  typedef Fft<const_Vector, cscalar_f, cscalar_f, fft_inv> i_fft_type;

  // Create FFT objects
  length_type N = 2048;
  f_fft_type f_fft(Domain<1>(N), 1.0);
  i_fft_type i_fft(Domain<1>(N), 1.0 / N);

  // Allocate input and output buffers
  Vector<cscalar_f> in(N, cscalar_f(1.f));
  Vector<cscalar_f> inv(N);
  Vector<cscalar_f> out(N);
  Vector<cscalar_f> ref(N, cscalar_f());
  ref(0) = cscalar_f(N);
  
  // Compute forward and inverse FFT's
  out = f_fft(in);
  inv = i_fft(out);
  
  // Validate the results (allowing for small numerical errors)
  test_assert(vsip_csl::error_db(ref, out) < -100);
  test_assert(vsip_csl::error_db(inv, in) < -100);

  std::cout << "forward fft mops: " << f_fft.impl_performance("mops") << std::endl;
  std::cout << "inverse fft mops: " << i_fft.impl_performance("mops") << std::endl;
}

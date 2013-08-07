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

/// Description: VSIPL++ Library: IIR Filter Example
///
/// This example demonstrates the use of VSIPL++ IIR
/// functionality by implementing a simple 2nd order
/// low-pass Butterworth filter and computing the step
/// response.

#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip/initfin.hpp>
#include <iostream>

using namespace vsip;

// =================================================================
// Matrices 'b' and 'a' are used to store the coefficients according 
// to the following definition relating the input vector 'x' to the 
// output vector 'y':
//
//                      k                   k
//              y[n] = sum (b[i]*x[n-i]) - sum(a[j]*y[n-j])
//                     i=0                 j=1
//
// where 'k' is the order of the filter.  This notation is used for
// the remainder of this example.  The computation will be 
// performed in single precision.

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // Set the order of the filter 'k' to 2.
  int k = 2;

  // Compute 20 discrete samples of the output from 20 input samples.
  length_type len = 20;

  // Matrices 'a' and 'b' to hold the coefficients for the delayed 
  // outputs and inputs respectively.
  Matrix<scalar_f> a(1, k);
  Matrix<scalar_f> b(1, k+1);

  // Vectors 'x' and 'y' to hold the input and output samples
  // respectively.  'x' is initialized to a unit step input.
  // 'y' is initialized to 0.
  Vector<scalar_f> x(len, 1.0F);
  Vector<scalar_f> y(len, 0.0F);

  // Fill 'a' and 'b' with the filter coefficients, in this case to
  // implement a simple low pass filter.
  b(0, 0) = 0.3913F;
  b(0, 1) = 0.7827F;
  b(0, 2) = 0.3913F;

  a(0, 0) = 0.3695F;
  a(0, 1) = 0.1958F;

  // Setup an IIR object to use the coefficients in 'b' and 'a' and 
  // to operate on vectors of length 'len'.  'state_no_save' indicates
  // that the IIR filter object will not save its weighted combination
  // of delayed inputs and output between calls.  'state_save' will
  // allow the iir object to save the final state of it's previous
  // call to initialize it's subsequent calls.  
  // Perform the filter using 'state_no_save' as a single input and then
  // perform the filter as two inputs in two separate operations yet
  // use the 'state_save' option to retain the filter state between
  // calls.
  Iir<scalar_f, state_no_save> iir_ns(b, a, len);
  Iir<scalar_f, state_save> iir_s(b, a, len/2);
  
  // Compute the step response for the entire input at once using
  // the 'state_no_save' object.  The input is in 'x' and the result
  // is placed in 'y'.
  iir_ns(x, y);

  // View the output using the get accessor to examine vector data
  for(index_type i = 0; i < len; ++i)
  {
    std::cout << y.get(i) << " ";
  }
  std::cout << std::endl;

  // Compute the step response in two stages using the first half
  // of the input first followed by the second half and retaining
  // the filter state between calls.  For the purposes of this
  // example the IIR object will be called on two subsets of the
  // input, the first half and the second half.
  iir_s(x(Domain<1>(0, 1, len/2)), y(Domain<1>(0, 1, len/2)));
  iir_s(x(Domain<1>(len/2, 1, len/2)), y(Domain<1>(len/2, 1, len/2)));

  // View the output of the two stage IIR filter for comparison
  // to the single stage IIR, the two should be the same by virtue
  // of saving the state between calls.
  for(index_type i = 0; i < len; ++i)
  {
    std::cout << y.get(i) << " ";
  }
  std::cout << std::endl;
}

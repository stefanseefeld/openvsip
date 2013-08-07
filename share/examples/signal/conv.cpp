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

/// Description: VSIPL++ Library: Convolution/Fast Convolution Example
///
/// This example demonstrates the use of VSIPL++ time domain and frequency
/// domain convolution functionality.

#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip/initfin.hpp>
#include <vsip/selgen.hpp>
#include <ovxx/output.hpp>
#include <iostream>

using namespace vsip;

// ============================================================================
// Vectors 'input', 'output_*', and 'kernel' are used to store the input,
//  output for various support sizes, and coefficients respectively.  The output
//  will be computed via time domain convolution using the support regions
//  defined by VSIPL++ and also via circular convolution in the frequency
//  domain, also known as fast convolution.
int
main(int argc, char** argv)
{
  typedef complex<float> T;

  vsipl init(argc, argv);

  // Set the decimation to one
  static length_type const dec = 1;

  length_type in_len = 10; // Input length
  length_type krn_len = 4; // Kernel length

  // The different support sizes for convolution define how the edge conditions
  //  are handled.  In all cases of linear convolution the edges are zero
  //  padded, the only difference is how many elements on either side of the
  //  input are added.  For a decimation of one the full support size will pad
  //  the input with exactly krn_len - 1 elements on either side.  This provides
  //  the maximum length output for a given input and kernel.  
  //  For the same support size (also for a decimation of 1) the padding is
  //  chosen such that the output length is identical to the input length.  This
  //  is described in the VSIPL++ specification, in short, it is done in such
  //  a way that there is equal amount of padding on either side of the input
  //  and is taken to be krn_len / 2 elements.  If there must be a different
  //  number of elements on one side then the larger amount of padding is taken
  //  on the beginning of the input.
  //  Finally, for the minimum support case there is no padding used and the 
  //  output length is the smallest length which allows the kernel to cover the
  //  full range of the input.

  // Output length for:
  //  Full support
  length_type out_len_full = (in_len + krn_len - 2) / dec + 1;
  //  Same support
  length_type out_len_same = (in_len - 1) / dec + 1;
  //  Minimum support
  length_type out_len_min = (in_len - 1) / dec - (krn_len - 1) / dec + 1;

  // Create the input, outputs, and coefficients
  Vector<T> input(in_len);
  Vector<T> kernel(krn_len, T(1));

  Vector<T> output_full(out_len_full);
  Vector<T> output_same(out_len_same);
  Vector<T> output_min(out_len_min);

  // No special symmetry applies to the coefficient vector
  symmetry_type const symm = nonsym;
  
  // Set up a ramp for the input.
  input = ramp(T(), T(1), in_len);

  // Create a convolution object for each support type
  Convolution<const_Vector, symm, support_full, T>
              conv_full(kernel, Domain<1>(in_len), dec);
  Convolution<const_Vector, symm, support_same, T>
              conv_same(kernel, Domain<1>(in_len), dec);
  Convolution<const_Vector, symm, support_min, T>
              conv_min(kernel, Domain<1>(in_len), dec);

  // Perform the convolutions
  conv_full(input, output_full);
  conv_same(input, output_same);
  conv_min(input, output_min);

  std::cout << "\n Illustration of convolution support regions\n"
            << " Elements in brackets [] are padding elements\n";
  std::cout << "\nFull support time domain convolution:\n";

  std::cout << "[(0,0) (0,0) (0,0)]";
  for (int i = 0; i < in_len - 1; ++i)
    std::cout << input(i) << " ";

  std::cout << input(in_len - 1) << "[(0,0) (0,0) (0,0)]\n ";

  for (int i = 0; i < krn_len; ++i)
    std::cout << kernel(i) << " ";

  std::cout << "----------->";
  for (int i = 0; i < krn_len; ++i)
    std::cout << kernel(i) << " ";

  std::cout << "----------->";
  for (int i = 0; i < krn_len; ++i)
    std::cout << kernel(i) << " ";

  std::cout << "\n\n= ";
  for (int i = 0; i < out_len_full; ++i)
    std::cout << output_full(i) << " ";

  std::cout << "\n";
  std::cout << "\nSame support time domain convolution:\n";
  std::cout << "  [(0,0)]";
  for (int i = 0; i < in_len - 1; ++i)
    std::cout << input(i) << " ";

  std::cout << input(in_len - 1) << "[(0,0) (0,0)]\n ";

  std::cout << "  ";
  for (int i = 0; i < krn_len; ++i)
    std::cout << kernel(i) << " ";

  std::cout << "----------------------------->";
  for (int i = 0; i < krn_len; ++i)
    std::cout << kernel(i) << " ";

  std::cout << "\n\n=  ";
  for (int i = 0; i < out_len_same; ++i)
    std::cout << output_same(i) << " ";

  std::cout << "\n";
  std::cout << "\nMinimum support time domain convolution:\n     ";
  for (int i = 0; i < in_len; ++i)
    std::cout << input(i) << " ";

  std::cout << "\n     ";
  for (int i = 0; i < krn_len; ++i)
    std::cout << kernel(i) << " ";

  std::cout << "----------->";
  for (int i = 0; i < krn_len; ++i)
    std::cout << kernel(i) << " ";

  std::cout << "\n\n=    ";
  for (int i = 0; i < out_len_min; ++i)
    std::cout << output_min(i) << " ";

  std::cout << "\n";

//=============================================================================
//  Perform a frequency domain circular convolution
//   The circular convolution via frequency domain techniques will be computed
//   in a standard step by step fashion as well as a fused operation.  The
//   fused operation makes use of VSIPL++ expression templates to map the 
//   sequence of operations (FFT -> elementwise multiply -> inverse FFT) to a
//   single expression object.  In this way the intermediate temporary
//   calculations and storage can be reduced in order to improve performance.

  // Create a zero padded full length vector for the kernel coefficients
  Vector<T> kernel_padded(in_len);

  for (int i = 0; i < krn_len; ++i)
    kernel_padded(i) = kernel(i);

  // Create vectors for the intermediate results
  Vector<T > input_fft(in_len);
  Vector<T > kernel_fft(in_len);
  Vector<T > product(in_len);

  // Create a matrix for the input and output to do a fused fast convolution
  Matrix<T> input_matrix(1, in_len);
  Matrix<T> output_matrix(1, out_len_same);

  for (int i = 0; i < in_len; ++i)
    input_matrix(0, i) = input(i);

  // Create the forward and inverse FFTs
  Fft<const_Vector, cscalar_f, cscalar_f, fft_fwd>
      for_fft(Domain<1>(in_len), 1.0);
  Fft<const_Vector, T, T, fft_inv>
      inv_fft(Domain<1>(out_len_same), 1.0 / out_len_same);

  // Create forward and inverse FFTms for fused operation
  Fftm<T, T, row, fft_fwd, by_value, 0>
       for_fftm(Domain<2>(1, in_len), 1.0);
  Fftm<T, T, row, fft_inv, by_value, 0>
       inv_fftm(Domain<2>(1, out_len_same), 1.0 / out_len_same);

  // Compute the forward FFT of the input
  input_fft = for_fft(input);
  
  for (int i = 0; i < in_len / 2 - 1; ++i)
    input_fft(i + in_len / 2 + 1) = conj(input_fft(in_len / 2 - i - 1));

  // Compute the forward FFT of the kernel
  kernel_fft = for_fft(kernel_padded);

  for (int i = 0; i < in_len / 2 - 1; ++i)
    kernel_fft(i + in_len / 2 + 1) = conj(kernel_fft(in_len / 2 - i - 1));

  // Compute the product in the frequency domain
  product = input_fft * kernel_fft;

  // Compute the inverse FFT of the product
  output_same = inv_fft(product);

  // Compute the same result using a fused fast convolution by expressing all
  //  operations on a single line.
  output_matrix = inv_fftm(vmmul<0>(kernel_fft, for_fftm(input_matrix)));

  std::cout << "\nCircular convolution computed via frequency domain:\n= ";

  for (int i = 0; i < out_len_same; ++i)
    std::cout << output_same(i) << " ";

  std::cout << "\n\nFused computation of circular "
            << "convolution via frequency domain\n= ";

  for (int i = 0; i < out_len_same; ++i)
    std::cout << output_matrix(0, i) << " ";

  std::cout << "\n";

  // Reproducing circular convolution results via time domain convolution.
  //  A circular convolution can be computed as a minimum support convolution
  //  where the first krn_len - 1 elements of the input are the same as the
  //  last krn_len - 1 elements of the input.  This is essentially a same 
  //  support convolution with the original input length but with two
  //  differences: 1. The edges are padded with the beginning and end of the
  //  original input as if it were periodic and 2. The overlap (in this case)
  //  is entirely at the beginning of the input so that no padding is required
  //  on the end of the input.

  // Create a vector to hold the padded input
  Vector<T> input_padded(in_len + krn_len - 1);

  // Pad the input with it's end values
  for (int i = 0; i < krn_len - 1; ++i)
    input_padded(i) = input(in_len - krn_len + i + 1);

  // Fill in the rest of the input vector
  for (int i = krn_len - 1; i < in_len + krn_len - 1; ++i)
    input_padded(i) = input(i - krn_len + 1);

  // Create a convolution object to perform a minimum support convolution on
  //  the padded input
  Convolution<const_Vector, symm, support_min, T>
       conv_min_circ(kernel, Domain<1>(in_len + krn_len - 1), dec);

  // Compute the convolution
  conv_min_circ(input_padded, output_same);

  std::cout << "\nCircular convolution computed via time domain convolution:\n";

  std::cout << "[";
  for (int i = 0; i < krn_len - 2; ++i)
    std::cout << input_padded(i) << " ";

  std::cout << input_padded(krn_len - 2) << "]";
  for (int i = 0; i < in_len; ++i)
    std::cout << input(i) << " ";

  std::cout << "\n";

  std::cout << " ";
  for (int i = 0; i < krn_len; ++i)
    std::cout << kernel(i) << " ";

  std::cout << "----------------------------->";
  for (int i = 0; i < krn_len; ++i)
    std::cout << kernel(i) << " ";

  std::cout << "\n\n= ";
  for (int i = 0; i < out_len_same; ++i)
    std::cout << output_same(i) << " ";

  std::cout << "\n";
}

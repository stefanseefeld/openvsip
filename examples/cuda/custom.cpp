/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   Sourcery VSIPL++ example demonstrating how to write a custom
///   CUDA kernel that interoperates with SV++ views.

#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>


// Sourcery VSIPL++ extensions to the CUDA API include Direct Data Access 
// to a blocks' memory on the GPU (in global memory).  
#include <vsip_csl/cuda.hpp>

using namespace vsip;
namespace cuda = vsip_csl::cuda;


// The CUDA kernel is defined in custom_kernel.cu
extern void
copy(float const *input, ptrdiff_t in_r_stride, ptrdiff_t in_c_stride,
     float *output, ptrdiff_t out_r_stride, ptrdiff_t out_c_stride,
     size_t rows, size_t cols);


int
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  typedef scalar_f T;
  
  Matrix<T> input(4, 4, T());
  Matrix<T> output(4, 4, T());

  // Fill input with ramp
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      input(i, j) = i * 4. + j + 1;

  std::cout << "Creating input with values {1...16}" << std::endl
            << input << std::endl;


  // Data in GPU memory may be safely accessed while the cuda::dda::Data
  // objects are in scope.
  { 
    typedef Matrix<T>::block_type block_type;

    cuda::dda::Data<block_type, dda::in> in(input.block());
    cuda::dda::Data<block_type, dda::out> out(output.block());

    std::cout << "Calling CUDA copy kernel" << std::endl << std::endl;

    copy(
      in.ptr(), in.stride(0), in.stride(1),
      out.ptr(), out.stride(0), out.stride(1),
      in.size(0), in.size(1));
  }

  // It is now safe to access the data through the view again, as one
  // normally would.
  std::cout << "Expecting output to match input" << std::endl
            << output << std::endl;

  if (alltrue(input == output))
    std::cout << "Pass" << std::endl;
  else
    std::cout << "Fail" << std::endl;
}

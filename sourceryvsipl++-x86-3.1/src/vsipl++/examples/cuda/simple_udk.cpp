/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

#include <vsip/initfin.hpp>
#include <vsip_csl/udk.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <iostream>

using namespace vsip;
namespace udk = vsip_csl::udk;
namespace cuda = vsip::impl::cuda;


// This custom CUDA kernel is defined in simple_udk_cuda.cu
extern void
udk_copy(float const *input, ptrdiff_t in_r_stride, ptrdiff_t in_c_stride,
         float *output, ptrdiff_t out_r_stride, ptrdiff_t out_c_stride,
         size_t rows, size_t cols);

// This is a wrapper function whose signature is mandated by the Task
// it is bound to. It thus bridges between the Task and the CUDA kernel, by
// unwrapping the function arguments to be passed to the kernel.
void copy(cuda::dda::Data<Dense<2>, dda::in> &in,
	  cuda::dda::Data<Dense<2>, dda::out> &out)
{
  udk_copy(in.ptr(), in.stride(0), in.stride(1),
   	   out.ptr(), out.stride(0), out.stride(1),
   	   in.size(0), in.size(1));
}

int
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  
  // Create a CUDA Task that copies one dense matrix to
  // another dense matrix, using a custom "copy" kernel.
  // ''in<T>'' marks an input parameter, while ''out<T>''
  // is used to mark an output parameter. ''inout<T>'' would
  // be used for a two-way parameter.
  udk::Task<udk::target::cuda, 
    udk::tuple<udk::in<Dense<2> >, udk::out<Dense<2> > > > 
    task(copy);
  // Create two matrices, one filled with 2., the other with 1.
  Matrix<float> input(4, 4, 2.);
  Matrix<float> output(4, 4, 1.);
  // Execute the task and wait for completion.
  // The Task::execute signature is generated from the Task template
  // argument list. In this case, a ''const_Matrix<float>'' is expected
  // for the first (input) argument, and a ''Matrix<float>'' for the second
  // (output).
  task.execute(input, output);
  // Make sure the two matrices are equal.
  test_assert(vsip_csl::view_equal(input, output));
}

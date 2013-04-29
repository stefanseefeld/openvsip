/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// VSIPL++ Library: Unit tests for user-kernels using Tensors.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/math.hpp>
#include <math.h>

#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/ukernel/kernels/host/tensor_mul2.hpp>
#include <vsip_csl/test.hpp>


void
test_tensor_ukernel(
  vsip::length_type const M,
  vsip::length_type const N,
  vsip::length_type const P)
{
  typedef vsip::Tensor<float> tensor_type;

  tensor_type in(M, N, P, 0);
  tensor_type out(M, N, P, 0);

  // Each element of the input matrix contains it's index.
  for(uint i = 0; i < in.size(0); ++i)
    for(uint j = 0; j < in.size(1); ++j)
      for(uint k = 0; k < in.size(2); ++k)
        in.put(i, j, k, i * in.size(1) * in.size(2) +j * in.size(2) + k);

  // This creates the user kernel that multiplies a tensor by two in
  // an elementwise fashion.
  vsip_csl::ukernel::Tensor_mul2_proxy kobj;

  // This prepares the kernel and executes it.
  vsip_csl::ukernel::Ukernel<vsip_csl::ukernel::Tensor_mul2_proxy> kernel(kobj);
  kernel(in, out);


  // Verify the result.
  test_assert(vsip_csl::view_equal(out, in * 2));
}


int main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_tensor_ukernel(1, 2, 64);
  test_tensor_ukernel(4, 4, 128);
  test_tensor_ukernel(8, 4, 10);
  test_tensor_ukernel(4, 8, 12);

  // Because the last dimension cannot be subdivided, the caller
  // must ensure that the input size plus the output size is
  // less than the limit defined in vsip/opt/cbe/overlay_params.h,
  // which is presently 64KB on cell.  The maximum sub-block size
  // also depends on the streaming patterns defined for the user
  // kernel (defined in vsip_csl/ukernel/kernels/host/tensor_mul2.hpp).
  test_tensor_ukernel(4, 4, 512);    // one sub-block of maximum size
  test_tensor_ukernel(16, 16, 512);
}


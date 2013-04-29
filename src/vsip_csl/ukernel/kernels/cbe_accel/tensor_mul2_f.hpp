/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// VSIPL++ Library: User-kernel for passing tensors (used primarily 
/// for testing).

#ifndef VSIP_CSL_UKERNEL_KERNELS_CBE_ACCEL_TENSOR_MUL2_F_HPP
#define VSIP_CSL_UKERNEL_KERNELS_CBE_ACCEL_TENSOR_MUL2_F_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>


/***********************************************************************
  Definitions
***********************************************************************/

namespace uk = vsip_csl::ukernel;

struct Tensor_mul2_kernel
    : uk::Kernel<uk::tuple<>, uk::tuple<float *>, uk::tuple<float *> >
{
  void compute(float *in, float *out, Pinfo const &pin, Pinfo const &pout)
  {
    // These consistency checks are possible with Tensors because they
    // are always constrained to be dense.  Hence, the strides can be
    // computed directly from the sizes, except for the smallest stride,
    // which must be one.
    bool consistent =
      (pin.l_total_size == pin.l_size[2] * pin.l_size[1] * pin.l_size[0]) &&
      (pin.l_stride[0] == pin.l_size[2] * pin.l_size[1]) &&
      (pin.l_stride[1] == pin.l_size[2]) &&
      (pin.l_stride[2] == 1) &&
      (pin.dim == 3);

    if (!consistent)
    {
      // Returning a negative value here will trigger an error on the
      // host side when the data are verified.
      out[0] = -1.f;
    }
    else
    {
      // Multiply the input by two.  This ensures each input value is 
      // read and each output value written.
      for(int i = 0; i < pin.l_total_size; ++i)
        out[i] = 2 * in[i];
    }
  }
};

#endif // VSIP_CSL_UKERNEL_KERNELS_CBE_ACCEL_TENSOR_MUL2_F_HPP

/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef KERNELS_CBE_ACCEL_BOX1_F_HPP
#define KERNELS_CBE_ACCEL_BOX1_F_HPP

#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>
#include <kernels/box1_param.hpp>
#include <algorithm>

namespace example
{
struct Box1_kernel : Spu_kernel
{
  typedef float* in0_type;
  typedef float* out0_type;

  typedef Box1_params param_type;

  void init(param_type const &params)
  {
    overlap = params.overlap;
  }

  void compute(
    in0_type      in,
    out0_type     out,
    Pinfo const&  p_in,
    Pinfo const&  p_out)
  {
    int size   = p_in.l_total_size;
    int offset = p_in.l_offset[0];

#if 1
    for (int r = 0; r < size; ++r)
    {
      float sum = 0;

      int rstart = -std::min(overlap, p_in.o_leading[0] + r);
      int rend   =  std::min(overlap, size-r-1+p_in.o_trailing[0]) + 1;

      for (int rr = rstart; rr < rend; ++rr)
	sum += in[r+offset+rr];
      out[r] = sum;
    }
#else
    if (p_in.o_leading[0] > 0)
      out[0] = in[offset - 1] 
	     + in[offset] 
	     + in[offset + 1];
    else
      out[0] = in[offset] 
	     + in[offset + 1];

    for (int i = 1; i < length-1; ++i)
      out[i] = in[i + offset - 1] 
	     + in[i + offset] 
	     + in[i + offset + 1];

    if (p_in.o_trailing[0] > 0)
      out[length-1] = in[length-1 + offset - 1] 
	            + in[length-1 + offset] 
	            + in[length-1 + offset + 1];
    else
      out[length-1] = in[length-1 + offset] 
	            + in[length-1 + offset - 1];
#endif
  }

  int       overlap;
};
}
#endif

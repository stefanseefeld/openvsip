/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef KERNELS_CBE_ACCEL_BOX2_F_HPP
#define KERNELS_CBE_ACCEL_BOX2_F_HPP

#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>
#include <kernels/box2_param.hpp>
#include <algorithm>

namespace example
{
struct Box2_kernel : Spu_kernel
{
  typedef float* in0_type;
  typedef float* out0_type;

  typedef Box2_params param_type;

  void init(param_type const &params)
  {
    overlap0 = params.overlap0;
    overlap1 = params.overlap1;
  }

  void compute(in0_type      in,
	       out0_type     out,
	       Pinfo const&  p_in,
	       Pinfo const&  p_out)
  {
    int size0   = p_in.l_size[0];
    int size1   = p_in.l_size[1];
    int in_offset0 = p_in.l_offset[0];
    int in_offset1 = p_in.l_offset[1];
    int in_stride0 = p_in.l_stride[0];
    int in_stride1 = p_in.l_stride[1];

    int out_offset0 = p_out.l_offset[0];
    int out_offset1 = p_out.l_offset[1];
    int out_stride0 = p_out.l_stride[0];
    int out_stride1 = p_out.l_stride[1];

#if DEBUG
    printf("box2 compute:  size %d x %d  in_off: %d x %d  in_str: %d x %d\n",
	   size0, size1,
	   in_offset0, in_offset1,
	   in_stride0, in_stride1);
#endif

    if (overlap0 == 1 && overlap1 == 1 &&
	p_in.o_leading[0] > 0 &&
	p_in.o_trailing[0] > 0 &&
	p_in.o_leading[1] > 0 &&
	p_in.o_trailing[1] > 0)
    {
      for (int r = 0; r < size0; ++r)
	for (int c = 0; c < size1; ++c)
	{
	  out[r * out_stride0 + c] = 
	    in[(r+in_offset0-1) * in_stride0 + (c+in_offset1-1)] +
	    in[(r+in_offset0-1) * in_stride0 + (c+in_offset1+0)] +
	    in[(r+in_offset0-1) * in_stride0 + (c+in_offset1+1)] +
	    in[(r+in_offset0+0) * in_stride0 + (c+in_offset1-1)] +
	    in[(r+in_offset0+0) * in_stride0 + (c+in_offset1+0)] +
	    in[(r+in_offset0+0) * in_stride0 + (c+in_offset1+1)] +
	    in[(r+in_offset0+1) * in_stride0 + (c+in_offset1-1)] +
	    in[(r+in_offset0+1) * in_stride0 + (c+in_offset1+0)] +
	    in[(r+in_offset0+1) * in_stride0 + (c+in_offset1+1)];
	}
    }
    else
    {
      for (int r = 0; r < size0; ++r)
	for (int c = 0; c < size1; ++c)
	{
	  float sum = 0;

	  int rstart = -std::min(overlap0, p_in.o_leading[0] + r);
	  int cstart = -std::min(overlap1, p_in.o_leading[1] + c);
	  int rend   =  std::min(overlap0, size0-r-1+p_in.o_trailing[0]) + 1;
	  int cend   =  std::min(overlap1, size1-c-1+p_in.o_trailing[1]) + 1;
	  
	  for (int rr = rstart; rr < rend; ++rr)
	    for (int cc = cstart; cc < cend; ++cc)
	      sum +=
		in[(r+in_offset0+rr) * in_stride0 + (c+in_offset1+cc)];
	  out[r * out_stride0 + c] = sum;
	}
    }
  }

  int overlap0, overlap1;
};
}
#endif

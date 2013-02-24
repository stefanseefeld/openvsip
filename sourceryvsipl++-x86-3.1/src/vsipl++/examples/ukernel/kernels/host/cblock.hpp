/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    kernels/host/cblock.hpp
    @author  Jules Bergmann, Stefan Seefeld
    @date    2008-12-12
    @brief   VSIPL++ Library: User-defined kernel example illustrating
                              a control-block approach.
*/

#ifndef KERNELS_HOST_CBLOCK_HPP
#define KERNELS_HOST_CBLOCK_HPP

#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <kernels/cblock_params.hpp>

namespace example
{
// Host-side vector elementwise multiply-add ukernel.
class Cblock : public vsip_csl::ukernel::Host_kernel
{
  // Parameters:
  //  - in_argc and out_argc describe the number of input and output
  //    streams.
  //  - param_type describes the parameters sent to the accelerator.
public:
  static unsigned int const in_argc  = 0;
  static unsigned int const out_argc = 0;
  typedef Cblock_params param_type;

  // Host-side ukernel object initialization.
  //
  // Streaming pattern divides matrix into whole, single rows.
  Cblock(uint64_t in0, int32_t in0_stride,
	 uint64_t in1, int32_t in1_stride,
	 uint64_t in2, int32_t in2_stride,
	 uint64_t out, int32_t out_stride,
	 int32_t rows, int32_t cols)
    : in0_(in0), in0_stride_(in0_stride),
      in1_(in1), in1_stride_(in1_stride),
      in2_(in2), in2_stride_(in2_stride),
      out_(out), out_stride_(out_stride),
      rows_(rows), cols_(cols)
  {}

  // Host-side compute kernel.  Used if accelerator is not available.
  void compute() {}

  // Query API:
  // - fill_params() fills the parameter block to be passed to the
  //   accelerators.
  void fill_params(param_type& param) const
  {
    param.in0        = in0_;
    param.in0_stride = in0_stride_;
    param.in1        = in1_;
    param.in1_stride = in1_stride_;
    param.in2        = in2_;
    param.in2_stride = in2_stride_;
    param.out        = out_;
    param.out_stride = out_stride_;
    param.rows       = rows_;
    param.cols       = cols_;
  }

private:
  uint64_t in0_; int32_t in0_stride_;
  uint64_t in1_; int32_t in1_stride_;
  uint64_t in2_; int32_t in2_stride_;
  uint64_t out_; int32_t out_stride_;
  int32_t rows_; int32_t cols_;
};
}

#endif

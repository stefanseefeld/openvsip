/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

///Description: Standalone vmul ukernel.

#include <utility>
#include <cml.h>
#include <cml_core.h>
#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>

namespace example
{
namespace uk = vsip_csl::ukernel;

struct Vmul_kernel
  : uk::Kernel<uk::tuple<>, uk::tuple<float*,float*>, uk::tuple<float*> >
{
  void compute(float *in0, float *in1, float *out,
	       Pinfo const &p_in0, Pinfo const &p_in1, Pinfo const &p_out)
  {
    cml_vmul1_f(in0, in1, out, p_out.l_total_size);
  }
};
}

DEFINE_KERNEL(example::Vmul_kernel)

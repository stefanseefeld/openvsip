/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef KERNELS_CBE_ACCEL_ZVMMUL_HPP
#define KERNELS_CBE_ACCEL_ZVMMUL_HPP

#include <utility>
#include <complex>

#include <cml.h>
#include <cml_core.h>

#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>
#include <kernels/vmmul_param.hpp>

namespace example
{
namespace uk = vsip_csl::ukernel;

struct Zvmmul_kernel : uk::Kernel<uk::tuple<std::pair<float*, float*> >,
				  uk::tuple<std::pair<float*, float*> >,
				  uk::tuple<std::pair<float*, float*> >,
				  Vmmul_params>
{
  void pre_compute(std::pair<float*, float*> in, Pinfo const &p_in)
  {
    float* r = (float*)buf;
    float* i = (float*)buf + p_in.l_total_size;
    cml_core_vcopy1_f((float const*)in.first,  r, p_in.l_total_size);
    cml_core_vcopy1_f((float const*)in.second, i, p_in.l_total_size);
  }

  void compute(std::pair<float*, float*> in,
	       std::pair<float*, float*> out,
	       Pinfo const& p_in,
	       Pinfo const& p_out)
  {
    cml_zvmul1_f((float const*)buf,
		 (float const*)buf + p_out.l_total_size,
		 (float const*)in.first, (float const*)in.second,
		 (float*)out.first, (float*)out.second,
		 p_out.l_total_size);
  }

  vector float buf[2*4096/4];	// Use vector float to force alignment
};
} // namespace example

#endif

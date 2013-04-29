/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   Elementwise vector-matrix multiply ukernel.

#ifndef KERNELS_CBE_ACCEL_CVMMUL_HPP
#define KERNELS_CBE_ACCEL_CVMMUL_HPP

#include <utility>
#include <complex>

#include <cml.h>
#include <cml_core.h>

#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>
#include <kernels/vmmul_param.hpp>

namespace example
{
namespace uk = vsip_csl::ukernel;

struct Cvmmul_kernel : uk::Kernel<uk::tuple<std::complex<float> *>,
				  uk::tuple<std::complex<float> *>,
				  uk::tuple<std::complex<float> *>,
				  Vmmul_params>
{
  void pre_compute(std::complex<float> *in, Pinfo const &p_in)
  {
    cml_core_vcopy1_f((float const*)in, (float*)buf, 2*p_in.l_total_size);
  }

  void compute(std::complex<float> *in,
	       std::complex<float> *out,
	       Pinfo const& p_in,
	       Pinfo const& p_out)
  {
    cml_cvmul1_f((float const*)buf, (float const*)in,
		 (float*)out, p_out.l_total_size);
  }

  vector float buf[2*4096/4];	// Use vector float to force alignment
};
} // namespace example

#endif

/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef KERNELS_CBE_ACCEL_CVMUL_HPP
#define KERNELS_CBE_ACCEL_CVMUL_HPP

#include <utility>
#include <complex>
#include <cml.h>
#include <cml_core.h>
#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>

namespace example
{
struct Cvmul_kernel : Spu_kernel
{
  typedef std::complex<float>* in0_type;
  typedef std::complex<float>* in1_type;
  typedef std::complex<float>* out0_type;

  static unsigned int const in_argc  = 2;
  static unsigned int const out_argc = 1;

  static bool const in_place = true;

  void compute(in0_type in0,
	       in1_type in1,
	       out0_type out,
	       Pinfo const &p_in0,
	       Pinfo const &p_in1,
	       Pinfo const &p_out)
  {
    cml_cvmul1_f((float const*)in0, (float const*)in1,
		 (float*)out, p_out.l_total_size);
  }
};
} // namespace example

#endif

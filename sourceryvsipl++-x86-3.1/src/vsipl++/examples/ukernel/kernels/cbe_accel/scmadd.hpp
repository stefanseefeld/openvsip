/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef KERNELS_CBE_ACCEL_SCMADD_F_HPP
#define KERNELS_CBE_ACCEL_SCMADD_F_HPP

#include <utility>
#include <complex>

#include <cml.h>
#include <cml_core.h>

#include <vsip/opt/ukernel/cbe_accel/debug.hpp>
#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>

#define DEBUG 0

namespace example
{
struct Scmadd_kernel : Spu_kernel
{
  typedef float*               in0_type;
  typedef std::complex<float>* in1_type;
  typedef std::complex<float>* in2_type;
  typedef std::complex<float>* out0_type;

  static unsigned int const in_argc  = 3;
  static unsigned int const out_argc = 1;

  static bool const in_place = true;

  void compute(in0_type in0,
	       in1_type in1,
	       in2_type in2,
	       out0_type out,
	       Pinfo const &p_in0,
	       Pinfo const &p_in1,
	       Pinfo const &p_in2,
	       Pinfo const &p_out)
  {
#if DEBUG
    cbe_debug_dump_pinfo("p_in0", p_in0);
    cbe_debug_dump_pinfo("p_in1", p_in1);
    cbe_debug_dump_pinfo("p_in2", p_in2);
    cbe_debug_dump_pinfo("p_out", p_out);
#endif

    size_t size0 = p_in0.l_size[0];
    size_t size1 = p_in0.l_size[1];
    size_t stride = p_in0.l_stride[0];

    for (int i = 0; i < size0; ++i)
    {
      in0_type pi0 = &in0[i * stride];
      in1_type pi1 = &in1[i * stride];
      in2_type pi2 = &in2[i * stride];
      out0_type po = &out[i * stride];
   
      for (int j = 0; j < size1; ++j)
        po[j] = pi0[j] * pi1[j] + pi2[j];
    }
  }

};
}

#endif

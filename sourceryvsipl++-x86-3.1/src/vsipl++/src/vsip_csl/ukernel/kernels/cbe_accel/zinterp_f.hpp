/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ukernel/kernels/cbe_accel/zinterp_f.hpp
    @author  Don McCoy
    @date    2008-08-26
    @brief   VSIPL++ Library: User-defined polar to rectangular
               interpolation kernel for SSAR images.
*/
#ifndef VSIP_CSL_UKERNEL_KERNELS_CBE_ACCEL_ZINTERP_F_HPP
#define VSIP_CSL_UKERNEL_KERNELS_CBE_ACCEL_ZINTERP_F_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <stdio.h>

#include <spu_mfcio.h>
#include <utility>
#include <complex>
#include <cassert>

#include <cml.h>
#include <cml_core.h>

#include <vsip/opt/ukernel/cbe_accel/debug.hpp>
#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>

#define DEBUG 0
#define USE_OPTMIZED_INTERP 1
#define VSIP_IMPL_SPU_LITERAL(_type_, ...) ((_type_){__VA_ARGS__})


/***********************************************************************
  Definitions
***********************************************************************/

struct Zinterp_kernel : Spu_kernel
{
  typedef unsigned int*             in0_type;
  typedef float*                    in1_type;
  typedef std::pair<float*, float*> in2_type;
  typedef std::pair<float*, float*> out0_type;

  static unsigned int const in_argc  = 3;
  static unsigned int const out_argc = 1;

  static bool const in_place = true;

  void compute(
    in0_type     in0,
    in1_type     in1,
    in2_type     in2,
    out0_type    out,
    Pinfo const& p_in0,
    Pinfo const& p_in1,
    Pinfo const& p_in2,
    Pinfo const& p_out)
  {
    assert(p_in0.l_size[0] == p_in1.l_size[0]);
    assert(p_in1.l_size[0] == p_in2.l_size[0]);
    assert(p_in0.l_size[1] == p_in1.l_size[1]);
    assert(p_in1.l_size[1] == p_in2.l_size[1]);
    assert(p_in0.l_stride[0] == p_in2.l_stride[0]);
    assert(p_in0.l_stride[1] == p_in2.l_stride[1]);

    size_t const size0 = p_in1.l_size[0];
    size_t const size1 = p_in1.l_size[1];
    size_t const size2 = p_in1.l_size[2];
    size_t const act_size2 = 17;
    size_t const stride = p_in0.l_stride[0];
    size_t const l_total_size = p_out.l_total_size;

    float *in2_real = in2.first;
    float *in2_imag = in2.second;
    
    float *out_real = out.first;
    float *out_imag = out.second;

    // Reference (non-vectorized, non-unrolled) version of interp
    // algorithm.
    for (size_t i = 0; i < l_total_size; ++i)
    {
      out_real[i] = 0.0;
      out_imag[i] = 0.0;
    }

    for (size_t j = 0; j < size1; ++j)
    {
      size_t ikxrows = in0[j];
      size_t j_shift = (j + size1/2) % size1;
      for (size_t h = 0; h < act_size2; ++h)
      {
        out_real[ikxrows + h] += in2_real[j_shift] * in1[j * size2 + h];
        out_imag[ikxrows + h] += in2_imag[j_shift] * in1[j * size2 + h];
      }
    }

    for (size_t j = p_out.g_offset[0] % 2; j < l_total_size; j+=2)
    {
      out_real[j] *= -1.0;
      out_imag[j] *= -1.0;
    }

  }
};

#endif

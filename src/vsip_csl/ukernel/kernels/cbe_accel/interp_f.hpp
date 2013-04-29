/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ukernel/kernels/cbe_accel/interp_f.hpp
    @author  Don McCoy
    @date    2008-08-26
    @brief   VSIPL++ Library: User-defined polar to rectangular
               interpolation kernel for SSAR images.
*/
#ifndef VSIP_CSL_UKERNEL_KERNELS_CBE_ACCEL_INTERP_F_HPP
#define VSIP_CSL_UKERNEL_KERNELS_CBE_ACCEL_INTERP_F_HPP

/***********************************************************************
  Included Files
***********************************************************************/

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

struct Interp_kernel : Spu_kernel
{
  typedef unsigned int*        in0_type;
  typedef float*               in1_type;
  typedef std::complex<float>* in2_type;
  typedef std::complex<float>* out0_type;

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
#if DEBUG
    cbe_debug_dump_pinfo("p_in0", p_in0);
    cbe_debug_dump_pinfo("p_in1", p_in1);
    cbe_debug_dump_pinfo("p_in2", p_in2);
    cbe_debug_dump_pinfo("p_out", p_out);
#endif

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

    float* fout = (float*)out;

    vector float zero = ((vector float){0.f, 0.f, 0.f, 0.f});

#if USE_OPTMIZED_INTERP
    memset((void*)out, 0, l_total_size * sizeof(std::complex<float>));

// SIMD version
    vector unsigned char shuf_0011 = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    0,  1,  2,  3,  0,  1,  2,  3,
			    4,  5,  6,  7,  4,  5,  6,  7);
    vector unsigned char shuf_2233 = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    8,  9, 10, 11,  8,  9, 10, 11,
			   12, 13, 14, 15, 12, 13, 14, 15);
    vector unsigned char shuf_1122 = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    4,  5,  6,  7,  4,  5,  6,  7,
			    8,  9, 10, 11,  8,  9, 10, 11);
    vector unsigned char shuf_3344 = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			   12, 13, 14, 15, 12, 13, 14, 15,
			   16, 17, 18, 19, 16, 17, 18, 19);
    vector unsigned char shuf_0044 = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    0,  1,  2,  3,  0,  1,  2,  3,
			   16, 17, 18, 19, 16, 17, 18, 19);
    vector unsigned char shuf_0101 = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    0,  1,  2,  3,  4,  5,  6,  7,
			    0,  1,  2,  3,  4,  5,  6,  7);
    vector unsigned char shuf_2323 = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    8,  9, 10, 11, 12, 13, 14, 15,
			    8,  9, 10, 11, 12, 13, 14, 15);

    for (size_t j = 0; j < size1; ++j)
    {
      size_t j_shift = (j + size1/2) % size1;
      vector float vscale = *(vector float*)(&in2[j_shift]);
      if (j_shift % 2 == 0)
	vscale = spu_shuffle(vscale, vscale, shuf_0101);
      else
	vscale = spu_shuffle(vscale, vscale, shuf_2323);
      unsigned int const ui = 2*in0[j];

      if (ui % 4 == 0)
      {
	vector float* vf    = (vector float*)(&fout[ui]);
	vector float* psync = (vector float*)(&in1[j*size2]);

	for (size_t k = 0; k < act_size2-15; k += 16)
	{
	  vector float f0 = vf[0];
	  vector float f1 = vf[1];
	  vector float f2 = vf[2];
	  vector float f3 = vf[3];
	  vector float f4 = vf[4];
	  vector float f5 = vf[5];
	  vector float f6 = vf[6];
	  vector float f7 = vf[7];
	  vector float sync0123  = psync[0];
	  vector float sync4567  = psync[1];
	  vector float sync89ab  = psync[2];
	  vector float synccdef  = psync[3];
	  vector float sync0011 = spu_shuffle(sync0123, sync0123, shuf_0011);
	  vector float sync2233 = spu_shuffle(sync0123, sync0123, shuf_2233);
	  vector float sync4455 = spu_shuffle(sync4567, sync4567, shuf_0011);
	  vector float sync6677 = spu_shuffle(sync4567, sync4567, shuf_2233);
	  vector float sync8899 = spu_shuffle(sync89ab, sync89ab, shuf_0011);
	  vector float syncaabb = spu_shuffle(sync89ab, sync89ab, shuf_2233);
	  vector float syncccdd = spu_shuffle(synccdef, synccdef, shuf_0011);
	  vector float synceeff = spu_shuffle(synccdef, synccdef, shuf_2233);
	  f0 = spu_madd(vscale, sync0011, f0);
	  f1 = spu_madd(vscale, sync2233, f1);
	  f2 = spu_madd(vscale, sync4455, f2);
	  f3 = spu_madd(vscale, sync6677, f3);
	  f4 = spu_madd(vscale, sync8899, f4);
	  f5 = spu_madd(vscale, syncaabb, f5);
	  f6 = spu_madd(vscale, syncccdd, f6);
	  f7 = spu_madd(vscale, synceeff, f7);
	  vf[0] = f0;
	  vf[1] = f1;
	  vf[2] = f2;
	  vf[3] = f3;
	  vf[4] = f4;
	  vf[5] = f5;
	  vf[6] = f6;
	  vf[7] = f7;
	  vf += 8; psync += 4;
	}
	vector float f0 = vf[0];
	vector float sync0xxx = psync[0];
	vector float sync00xx = spu_shuffle(sync0xxx, zero, shuf_0044);
	f0 = spu_madd(vscale, sync00xx, f0);
	vf[0] = f0;
      }
      else
      {
	vector float* vf   = (vector float*)(&fout[ui + 2]);
	vector float* psync = (vector float*)(&in1[j*size2]);

	vector float sync0123 = psync[0];

	vector float f0 = vf[-1];
	vector float syncxx00 = spu_shuffle(zero, sync0123, shuf_0044);
	f0 = spu_madd(vscale, syncxx00, f0);
	vf[-1] = f0;

	for (size_t k = 1; k < act_size2-15; k += 16)
	{
	  vector float f0 = vf[0];
	  vector float f1 = vf[1];
	  vector float f2 = vf[2];
	  vector float f3 = vf[3];
	  vector float f4 = vf[4];
	  vector float f5 = vf[5];
	  vector float f6 = vf[6];
	  vector float f7 = vf[7];
	  vector float sync4567 = psync[1];
	  vector float sync89ab = psync[2];
	  vector float synccdef = psync[3];
	  vector float syncghij = psync[4];
	  vector float sync1122 = spu_shuffle(sync0123, sync4567, shuf_1122);
	  vector float sync3344 = spu_shuffle(sync0123, sync4567, shuf_3344);
	  vector float sync5566 = spu_shuffle(sync4567, sync89ab, shuf_1122);
	  vector float sync7788 = spu_shuffle(sync4567, sync89ab, shuf_3344);
	  vector float sync99aa = spu_shuffle(sync89ab, synccdef, shuf_1122);
	  vector float syncbbcc = spu_shuffle(sync89ab, synccdef, shuf_3344);
	  vector float syncddee = spu_shuffle(synccdef, syncghij, shuf_1122);
	  vector float syncffgg = spu_shuffle(synccdef, syncghij, shuf_3344);
	  sync0123 = syncghij;
	  f0 = spu_madd(vscale, sync1122, f0);
	  f1 = spu_madd(vscale, sync3344, f1);
	  f2 = spu_madd(vscale, sync5566, f2);
	  f3 = spu_madd(vscale, sync7788, f3);
	  f4 = spu_madd(vscale, sync99aa, f4);
	  f5 = spu_madd(vscale, syncbbcc, f5);
	  f6 = spu_madd(vscale, syncddee, f6);
	  f7 = spu_madd(vscale, syncffgg, f7);
	  vf[0] = f0;
	  vf[1] = f1;
	  vf[2] = f2;
	  vf[3] = f3;
	  vf[4] = f4;
	  vf[5] = f5;
	  vf[6] = f6;
	  vf[7] = f7;
	  vf += 8; psync += 4;
	}
      }
    }

    vector float scale;
    if (p_out.g_offset[0] % 2)
      scale = ((vector float){-1.f, -1.f, 1.f, 1.f});
    else
      scale = ((vector float){1.f, 1.f, -1.f, -1.f});

    vector float* vout = (vector float*)fout;
    for (size_t j = 0; j < l_total_size>>1; j++)
      vout[j] = spu_madd(vout[j], scale, zero);

#else
    // Reference (non-vectorized, non-unrolled) version of interp
    // algorithm.
    for (size_t i = 0; i < l_total_size; ++i)
      out[i] = std::complex<float>();

    for (size_t j = 0; j < size1; ++j)
    {
      size_t ikxrows = in0[j];
      size_t j_shift = (j + size1/2) % size1;
      for (size_t h = 0; h < act_size2; ++h)
        out[ikxrows + h] += in2[j_shift] * in1[j * size2 + h];
    }

    for (size_t j = p_out.g_offset[0] % 2; j < l_total_size; j+=2)
      out[j] *= -1.0;
#endif
  }

};

#endif

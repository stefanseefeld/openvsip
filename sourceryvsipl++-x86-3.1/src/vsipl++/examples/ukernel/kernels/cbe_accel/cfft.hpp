/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef KERNELS_CBE_ACCEL_CFFT_HPP
#define KERNELS_CBE_ACCEL_CFFT_HPP

#include <utility>
#include <complex>
#include <cassert>
#include <spu_intrinsics.h>
#include <cml.h>
#include <cml_core.h>
#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>
#include <kernels/fft_param.hpp>

#define MIN_FFT_1D_SIZE	  32
#define MAX_FFT_1D_SIZE	  4096

#define FFT_BUF1_SIZE_BYTES (2*MAX_FFT_1D_SIZE*sizeof(float))
#define FFT_BUF2_SIZE_BYTES (1*MAX_FFT_1D_SIZE*sizeof(float)+128)

namespace example
{
namespace uk = vsip_csl::ukernel;

struct Cfft_kernel : uk::Kernel<uk::tuple<>,
				uk::tuple<std::complex<float>*>,
				uk::tuple<std::complex<float>*>,
				Fft_params>
{
  void init(param_type const &params)
  {
    size  = params.size;
    dir   = params.dir;
    scale = params.scale;

    int rt = cml_fft1d_setup_f(&fft, CML_FFT_CC, size, buf2);
    assert(rt && fft);
  }

  void compute(std::complex<float> const *in,
	       std::complex<float> *out,
	       Pinfo const &p_in,
	       Pinfo const &p_out)
  {
    // Handle inverse FFT explicitly so that shuffle and scale can happen
    // in single step.
    cml_core_ccfft1d_op_mi_f(fft, (float*)in, (float*)out, CML_FFT_FWD);

    if (dir == -1)
    {
      if (scale != 1.f)
	cml_core_rcsvmul1_f(scale, (float*)out, (float*)out, size);
    }
    else
    {
      // Code for the inverse FFT taken from the CBE SDK Libraries
      // Overview and Users Guide, sec. 8.1.
      int const vec_size = 4;
      vector float* start = (vector float*)out;
      vector float* end   = start + 2 * size / vec_size;
      vector float  s0, s1, e0, e1;
      vector unsigned int mask = (vector unsigned int){-1, -1, 0, 0};
      vector float vscale = spu_splats(scale);
      unsigned int i;
      
      // Scale the output vector and swap the order of the outputs.
      // Note: there are two float values for each of 'n' complex values.
      s0 = e1 = *start;
      for (i = 0; i < size / vec_size; ++i) 
      {
	s1 = *(start + 1);
	e0 = *(--end);
	
	*start++ = spu_mul(spu_sel(e0, e1, mask), vscale);
	*end     = spu_mul(spu_sel(s0, s1, mask), vscale);
	s0 = s1;
	e1 = e0;
      }
    }
  }

  size_t      size;
  int         dir;
  float       scale;
  fft1d_f*    fft;

  static char buf1[FFT_BUF1_SIZE_BYTES];
  static char buf2[FFT_BUF2_SIZE_BYTES];
};
} // namespace example

#endif

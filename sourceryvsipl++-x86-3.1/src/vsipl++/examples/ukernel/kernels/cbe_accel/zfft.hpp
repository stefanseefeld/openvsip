/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef KERNELS_CBE_ACCEL_ZFFT_HPP
#define KERNELS_CBE_ACCEL_ZFFT_HPP

#include <utility>
#include <cassert>
#include <spu_intrinsics.h>

#include <cml.h>
#include <cml_core.h>
#include <vsip_csl/ukernel/cbe_accel/ukernel.hpp>
#include <kernels/fft_param.hpp>

#define MIN_FFT_1D_SIZE	  32
#define MAX_FFT_1D_SIZE	  4096

namespace example
{
namespace uk = vsip_csl::ukernel;

struct Zfft_kernel : uk::Kernel<uk::tuple<>,
				uk::tuple<std::pair<float*, float*> >,
				uk::tuple<std::pair<float*, float*> >,
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

  void compute(std::pair<float*,float*> const in,
	       std::pair<float*,float*> out,
	       Pinfo const &p_in,
	       Pinfo const &p_out)
  {
    cml_zzfft1d_op_f(fft,
		     in.first, in.second,
		     out.first, out.second,
		     dir, (float*)buf1);

    if (scale != 1.f)
      cml_core_rzsvmul1_f(scale, out.first, out.second,
			  out.first, out.second, size);
  }

  size_t      size;
  int         dir;
  float       scale;
  fft1d_f*    fft;

  static char buf1[2*MAX_FFT_1D_SIZE*sizeof(float)];
  static char buf2[1*MAX_FFT_1D_SIZE*sizeof(float)+128];
};
} // namespace example

#endif

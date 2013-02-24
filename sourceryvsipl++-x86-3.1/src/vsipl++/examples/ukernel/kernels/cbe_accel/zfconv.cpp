/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#include <kernels/cbe_accel/zfft.hpp>
#include <kernels/cbe_accel/zvmmul.hpp>
#include <kernels/fconv_params.hpp>

namespace example
{
struct Zfconv_kernel : uk::Kernel<uk::tuple<std::pair<float*, float*> >,
				  uk::tuple<std::pair<float*, float*> >,
				  uk::tuple<std::pair<float*, float*> >,
				  Fconv_params>
{
  void init(param_type const &params)
  {
    fwd_fft.init(params.fwd_fft_params);
    inv_fft.init(params.inv_fft_params);
  }

  void pre_compute(std::pair<float*, float*> in, Pinfo const &p_in)
  {
    vmmul.pre_compute(in, p_in);
  }

  void compute(std::pair<float*, float*> in,
	       std::pair<float*, float*> out,
	       Pinfo const &p_in,
	       Pinfo const &p_out)
  {
    fwd_fft.compute(in, out, p_in, p_in);
    // This clutters the input buffer.
    vmmul.compute(out, in, p_in, p_in);
    inv_fft.compute(in, out, p_in, p_out);
  }

  Zfft_kernel fwd_fft;
  Zvmmul_kernel vmmul;
  Zfft_kernel inv_fft;
};

char Zfft_kernel::buf1[2*MAX_FFT_1D_SIZE*sizeof(float)]
  __attribute((aligned(128)));
char Zfft_kernel::buf2[1*MAX_FFT_1D_SIZE*sizeof(float)+128]
  __attribute((aligned(128)));
}

DEFINE_KERNEL(example::Zfconv_kernel)

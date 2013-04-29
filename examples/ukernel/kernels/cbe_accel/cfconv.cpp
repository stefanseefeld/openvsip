/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#include <kernels/cbe_accel/cfft.hpp>
#include <kernels/cbe_accel/cvmmul.hpp>
#include <kernels/fconv_params.hpp>

namespace example
{
struct Cfconv_kernel : uk::Kernel<uk::tuple<std::complex<float> *>,
				  uk::tuple<std::complex<float> *>,
				  uk::tuple<std::complex<float> *>,
				  Fconv_params>
{
  void init(param_type const &params)
  {
    fwd_fft.init(params.fwd_fft_params);
    inv_fft.init(params.inv_fft_params);
  }

  void pre_compute(std::complex<float> *in, Pinfo const &p_in)
  {
    vmmul.pre_compute(in, p_in);
  }

  void compute(std::complex<float> *in,
	       std::complex<float> *out,
	       Pinfo const &p_in,
	       Pinfo const &p_out)
  {
    fwd_fft.compute(in, out, p_in, p_in);
    // This clutters the input buffer.
    vmmul.compute(out, in, p_in, p_in);
    inv_fft.compute(in, out, p_in, p_out);
  }

  Cfft_kernel fwd_fft;
  Cvmmul_kernel vmmul;
  Cfft_kernel inv_fft;
};

char Cfft_kernel::buf1[FFT_BUF1_SIZE_BYTES] __attribute((aligned(128)));
char Cfft_kernel::buf2[FFT_BUF2_SIZE_BYTES] __attribute((aligned(128)));

}

DEFINE_KERNEL(example::Cfconv_kernel)

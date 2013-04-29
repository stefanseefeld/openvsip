/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef KERNELS_PARAMS_FCONV_PARAMS_HPP
#define KERNELS_PARAMS_FCONV_PARAMS_HPP

#include <kernels/fft_param.hpp>
#include <kernels/vmmul_param.hpp>

namespace example
{
struct Fconv_params
{
  Fft_params fwd_fft_params;
  Vmmul_params vmmul_params;
  Fft_params inv_fft_params;
};
} // namespace example

#endif

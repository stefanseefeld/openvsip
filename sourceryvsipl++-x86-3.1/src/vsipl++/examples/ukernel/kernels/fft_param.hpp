/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef KERNELS_PARAMS_FFT_PARAM_HPP
#define KERNELS_PARAMS_FFT_PARAM_HPP

namespace example
{
struct Fft_params
{
  int           dir; // -1 forward, +1 inverse
  unsigned int  size;
  float         scale;
};
} // namespace example

#endif

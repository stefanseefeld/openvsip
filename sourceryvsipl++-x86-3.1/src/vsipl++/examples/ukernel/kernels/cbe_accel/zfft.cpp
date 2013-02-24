/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include "zfft.hpp"

namespace example
{
char Zfft_kernel::buf1[2*MAX_FFT_1D_SIZE*sizeof(float)]
  __attribute((aligned(128)));
char Zfft_kernel::buf2[1*MAX_FFT_1D_SIZE*sizeof(float)+128]
  __attribute((aligned(128)));
}

DEFINE_KERNEL(example::Zfft_kernel)

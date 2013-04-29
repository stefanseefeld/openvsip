/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include "cfft.hpp"

namespace example
{
char Cfft_kernel::buf1[FFT_BUF1_SIZE_BYTES] __attribute((aligned(128)));
char Cfft_kernel::buf2[FFT_BUF2_SIZE_BYTES] __attribute((aligned(128)));
}

DEFINE_KERNEL(example::Cfft_kernel)

/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    fft-common.hpp
    @author  Jules Bergmann
    @date    15 Jun 2005
    @brief   VSIPL++ Library: Common CSL FFT header.
*/

#ifndef CSL_FFT_COMMON_HPP
#define CSL_FFT_COMMON_HPP

struct Forward { static int const isign = -1; };
struct Inverse { static int const isign = +1; };

#endif // CSL_FFT_COMMON_HPP

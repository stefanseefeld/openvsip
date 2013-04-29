/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_CSL_FFT_HPP
#define VSIP_CSL_FFT_HPP

#include <vsip/core/fft/backend.hpp>
#include <vsip/core/fft/util.hpp>

namespace vsip_csl
{
namespace fft
{
using vsip::impl::fft::Fft_backend;
using vsip::impl::fft::Fftm_backend;
  
using vsip::impl::fft::is_power_of_two;
using vsip::impl::fft::exponent;
using vsip::impl::fft::axis;
using vsip::impl::fft::io_size;
}
}

#endif

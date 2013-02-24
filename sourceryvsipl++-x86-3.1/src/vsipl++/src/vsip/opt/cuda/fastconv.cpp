/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/fastconv.cpp
    @author  Don McCoy
    @date    2009-03-22
    @brief   VSIPL++ Library: Wrapper for fast convolution using CUDA
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cufft.h>
#include <cuda_runtime.h>
#include <vsip/core/config.hpp>
#include <vsip/core/fns_scalar.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/static_assert.hpp>
#include <vsip/math.hpp>
#include <vsip/opt/cuda/fastconv.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cuda
{

// Fast convolution binding for interleaved complex data.


template <dimension_type D, typename T, storage_format_type C>
void
Fastconv_base<D, T, C>::fconv
  (T const* in, T const* kernel, T* out, length_type rows, length_type columns, bool transform_kernel)
{
  // convert pointers to types the CUFFT library accepts
  typedef cufftComplex ctype;
  ctype* d_out = reinterpret_cast<ctype*>(out);
  ctype* d_kernel = const_cast<ctype*>(reinterpret_cast<ctype const*>(kernel));
  ctype* d_in = const_cast<ctype*>(reinterpret_cast<ctype const*>(in));

  cufftHandle plan;
  if (transform_kernel)
  {
    // Create a 1D FFT plan and transform the kernel
    cufftPlan1d(&plan, columns, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_kernel, d_kernel, CUFFT_FORWARD);
    cufftDestroy(plan);
  }

  // Create a FFTM plan
  cufftPlan1d(&plan, columns, CUFFT_C2C, rows);

  // transform the data
  cufftExecC2C(plan, d_in, d_in, CUFFT_FORWARD);

  // convolve with kernel, combine with scaling needed for inverse FFT
  typedef typename impl::scalar_of<T>::type scalar_type;
  scalar_type scale = 1 / static_cast<scalar_type>(columns);
  if (D == 1)
    vmmuls_row(kernel, in, out, scale, rows, columns);
  else
    mmmuls(kernel, in, out, scale, rows, columns);

  // inverse transform the signal
  cufftExecC2C(plan, d_out, d_out, CUFFT_INVERSE);
  cufftDestroy(plan);
}



typedef std::complex<float> ctype;

template void
Fastconv_base<1, ctype, interleaved_complex>::fconv(
  ctype const* in, ctype const* kernel, ctype* out, 
  length_type rows, length_type columns, bool transform_kernel);

template void
Fastconv_base<2, ctype, interleaved_complex>::fconv(
  ctype const* in, ctype const* kernel, ctype* out, 
  length_type rows, length_type columns, bool transform_kernel);


          
} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

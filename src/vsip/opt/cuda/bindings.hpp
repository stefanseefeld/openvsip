/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/bindings.hpp
    @author  Don McCoy
    @date    2009-02-05
    @brief   VSIPL++ Library: Bindings for custom CUDA kernels.
*/

#ifndef VSIP_OPT_CUDA_BINDINGS_HPP
#define VSIP_OPT_CUDA_BINDINGS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/opt/cuda/device_storage.hpp>
#include <vsip/opt/cuda/kernels.hpp>

namespace vsip
{
namespace impl
{
namespace cuda
{

// Wrapper functions used for vmmul serial expression evaluator.
// These functions convert parameters to the proper (standard C) types
// and call the appropriate (non-overloaded) kernel entry point.

inline
void 
fftshift(float const*   input,
	 float*         output,
	 size_t         rows,
	 size_t         cols,
	 dimension_type in_major_dim, 
	 dimension_type out_major_dim)
{
  fftshift_s(input, output, rows, cols, in_major_dim, out_major_dim);
}

inline
void
fftshift(std::complex<float> const* input,
	 std::complex<float>*       output,
	 size_t                     rows,
	 size_t                     cols,
	 dimension_type             in_major_dim, 
	 dimension_type             out_major_dim)
{
  fftshift_c(reinterpret_cast<cuComplex const*>(input),
	     reinterpret_cast<cuComplex*>(output),
	     rows, cols, in_major_dim, out_major_dim);
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_CUDA_BINDINGS_HPP

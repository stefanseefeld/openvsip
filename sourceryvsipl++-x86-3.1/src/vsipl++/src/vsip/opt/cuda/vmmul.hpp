/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   vmmul cuda kernels

#ifndef vsip_opt_cuda_vmmul_hpp_
#define vsip_opt_cuda_vmmul_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>

namespace vsip
{
namespace impl
{
namespace cuda
{

/// Scalar-Matrix elementwise multiply, s * S --> S
void
smmul(float const  scale, float const* input, float *output,
      length_type rows, length_type cols);

/// Scalar-Matrix elementwise multiply, s * C --> C
void
smmul(float const scale,
      std::complex<float> const *input,
      std::complex<float> *output,
      length_type rows, length_type cols);

/// Vector-Matrix multiply, row-wise, S * S --> S
void 
vmmul_row(float const *kernel, float const *input, float *output,
	  length_type rows, length_type cols);

/// Vector-Matrix multiply, row-wise, S * C --> C
void 
vmmul_row(float const *kernel,
	  std::complex<float> const *input,
	  std::complex<float> *output,
	  length_type rows, length_type cols);

/// Vector-Matrix multiply, row-wise, C * S --> C
void 
vmmul_row(std::complex<float> const *kernel,  
	  float const *input,
	  std::complex<float> *output,
	  length_type rows, length_type cols);

/// Vector-Matrix multiply, row-wise, C * C --> C
void 
vmmul_row(std::complex<float> const *kernel,  
	  std::complex<float> const *input,
	  std::complex<float> *output,
	  length_type rows, length_type cols);

/// Vector-Matrix multiply, column-wise, S * S --> S
void 
vmmul_col(float const *kernel,  float const *input, float *output,
	  length_type rows, length_type cols);

/// Vector-Matrix multiply, column-wise, S * C --> C
void 
vmmul_col(float const *kernel,
	  std::complex<float> const *input,
	  std::complex<float> *output,
	  length_type rows, length_type cols);

/// Vector-Matrix multiply, column-wise, C * S --> C
void 
vmmul_col(std::complex<float> const *kernel,  
	  float const *input,
	  std::complex<float> *output,
	  length_type rows, length_type cols);

/// Vector-Matrix multiply, column-wise, C * C --> C
void 
vmmul_col(std::complex<float> const *kernel,  
	  std::complex<float> const *input,
	  std::complex<float> *output,
	  length_type rows, length_type cols);

/// Vector-Matrix multiply, with scale, row-wise, C * C --> C
void 
vmmuls_row(std::complex<float> const *kernel,  
	   std::complex<float> const *input,
	   std::complex<float> *output,
	   float scale, length_type rows, length_type cols);

/// Vector-Matrix multiply, with scale, column-wise, C * C --> C
void 
vmmuls_col(std::complex<float> const *kernel,  
	   std::complex<float> const *input,
	   std::complex<float> *output,
	   float scale, length_type rows, length_type cols);

/// Matrix-Matrix multiply, with scale, row-wise, C * C --> C
void 
mmmuls(std::complex<float> const *kernel,
       std::complex<float> const *input,
       std::complex<float> *output,
       float scale, length_type rows, length_type cols);

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif

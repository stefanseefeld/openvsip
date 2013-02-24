/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   Various device memory copy operations.

#ifndef vsip_opt_cuda_copy_hpp_
#define vsip_opt_cuda_copy_hpp_

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

/// dense scalar vector copy
void
copy(float const *in, float *out, length_type length);

/// dense complex vector copy
void
copy(complex<float> const *in, complex<float> *out, length_type length);

/// dense split-complex vector copy
inline void
copy(std::pair<float const*,float const*> in,
     std::pair<float*,float*> out, 
     length_type length)
{
  copy(in.first, out.first, length);
  copy(in.second, out.second, length);
}

/// scalar vector copy
void
copy(float const *in, stride_type in_stride,
     float *out, stride_type out_stride,
     length_type length);

/// complex vector copy
void
copy(complex<float> const *in, stride_type in_stride,
     complex<float> *out, stride_type out_stride,
     length_type length);

/// split-complex vector copy
inline void
copy(std::pair<float const*,float const*> in, stride_type in_stride,
     std::pair<float*,float*> out, stride_type out_stride,
     length_type length)
{
  copy(in.first, in_stride, out.first, out_stride, length);
  copy(in.second, in_stride, out.second, out_stride, length);
}

/// interleaved to split vector copy
inline void
copy(std::complex<float> const *in,
     std::pair<float*,float*> out,
     length_type length)
{
  copy(reinterpret_cast<float const*>(in), 2, out.first, 1, length);
  copy(reinterpret_cast<float const*>(in) + 1, 2, out.second, 1, length);
}

/// split to interleaved vector copy
inline void
copy(std::pair<float const *,float const *> in,
     std::complex<float> *out,
     length_type length)
{
  copy(in.first, 1, reinterpret_cast<float*>(out), 2, length);
  copy(in.second, 1, reinterpret_cast<float*>(out) + 1, 2, length);
}

/// dense float matrix copy
void
copy(float const *in, stride_type in_stride,
     float *out, stride_type out_stride,
     length_type rows, length_type cols);

/// dense complex matrix copy
void
copy(complex<float> const *in, stride_type in_stride,
     complex<float> *out, stride_type out_stride,
     length_type rows, length_type cols);

/// dense split-complex matrix copy
inline void
copy(std::pair<float const*,float const*> in, stride_type in_stride,
     std::pair<float*,float*> out, stride_type out_stride,
     length_type rows, length_type cols)
{
  copy(in.first, in_stride, out.first, out_stride, rows, cols);
  copy(in.second, in_stride, out.second, out_stride, rows, cols);
}

/// float matrix copy
void
copy(float const *in, stride_type in_stride_0, stride_type in_stride_1,
     float *out, stride_type out_stride_0, stride_type out_stride_1,
     length_type rows, length_type cols);

/// complex matrix copy
void
copy(complex<float> const *in, stride_type in_stride_0, stride_type in_stride_1,
     complex<float> *out, stride_type out_stride_0, stride_type out_stride_1,
     length_type rows, length_type cols);

/// interleaved to split matrix copy
inline void
copy(std::complex<float> const *in, stride_type in_stride,
     std::pair<float*,float*> out, stride_type out_stride,
     length_type rows, length_type cols)
{
  copy(reinterpret_cast<float const*>(in), 2 * in_stride, 2,
       out.first, out_stride, 1,
       rows, cols);
  copy(reinterpret_cast<float const*>(in) + 1, 2 * in_stride, 2,
       out.second, out_stride, 1,
       rows, cols);
}

/// split to interleaved matrix copy
inline void
copy(std::pair<float const *,float const *> in, stride_type in_stride,
     std::complex<float> *out, stride_type out_stride,
     length_type rows, length_type cols)
{
  copy(in.first, in_stride, 1,
       reinterpret_cast<float*>(out), 2 * out_stride, 2,
       rows, cols);
  copy(in.second, in_stride, 1,
       reinterpret_cast<float*>(out) + 1, 2 * out_stride, 2,
       rows, cols);
}

/// interleaved to split matrix copy
inline void
copy(std::complex<float> const *in, stride_type in_stride_0, stride_type in_stride_1,
     std::pair<float*,float*> out, stride_type out_stride_0, stride_type out_stride_1,
     length_type rows, length_type cols)
{
  copy(reinterpret_cast<float const*>(in), 2 * in_stride_0, 2 * in_stride_1,
       out.first, out_stride_0, out_stride_1,
       rows, cols);
  copy(reinterpret_cast<float const*>(in) + 1, 2 * in_stride_0, 2 * in_stride_1,
       out.second, out_stride_0, out_stride_1,
       rows, cols);
}

/// split to interleaved matrix copy
inline void
copy(std::pair<float const *,float const *> in,
     stride_type in_stride_0, stride_type in_stride_1,
     std::complex<float> *out, stride_type out_stride_0, stride_type out_stride_1,
     length_type rows, length_type cols)
{
  copy(in.first, in_stride_0, in_stride_1,
       reinterpret_cast<float*>(out), 2 * out_stride_0, 2 * out_stride_1,
       rows, cols);
  copy(in.second, in_stride_0, in_stride_1,
       reinterpret_cast<float*>(out) + 1, 2 * out_stride_0, 2 * out_stride_1,
       rows, cols);
}

/// dense float matrix transpose
void
transpose(float const *in, float *out,
	  length_type rows, length_type cols);

/// dense float matrix in-place transpose
void
transpose(float *inout, length_type size);

/// dense complex matrix transpose
void
transpose(complex<float> const *in, complex<float> *out,
	  length_type rows, length_type cols);

/// dense complex matrix in-place transpose
void
transpose(complex<float> *inout, length_type size);

/// dense split-complex matrix transpose
inline void
transpose(std::pair<float const*,float const*> in, 
	  std::pair<float*,float*> out,
	  length_type rows, length_type cols)
{
  transpose(in.first, out.first, rows, cols);
  transpose(in.second, out.second, rows, cols);
}

/// interleaved to split matrix transpose
inline void
transpose(complex<float> const *in, std::pair<float*,float*> out,
	  length_type rows, length_type cols)
{
  copy(reinterpret_cast<float const*>(in), 2*cols, 2, out.first, 1, rows,
       rows, cols);
  copy(reinterpret_cast<float const*>(in) + 1, 2*cols, 2, out.second, 1, rows,
       rows, cols);
}

/// split to interleaved matrix transpose
inline void
transpose(std::pair<float const*, float const*> in, complex<float> *out,
	  length_type rows, length_type cols)
{
  copy(in.first, cols, 1, reinterpret_cast<float*>(out), 2, 2*rows,
       rows, cols);
  copy(in.second, cols, 1, reinterpret_cast<float*>(out) + 1, 2, 2*rows,
       rows, cols);
}

/// scalar to vector assignment
void
assign_scalar(float value, float *out, length_type length);

/// complex to vector assignment
void
assign_scalar(std::complex<float> const &value,
	      std::complex<float> *out, 
	      length_type length);

/// complex to split vector assignment
inline void
assign_scalar(std::complex<float> const &value,
	      std::pair<float*,float*> out, 
	      length_type length)
{
  assign_scalar(value.real(), out.first, length);
  assign_scalar(value.imag(), out.second, length);
}

/// scalar to matrix assignment
void
assign_scalar(float value, float *out, stride_type stride,
	      length_type rows, length_type cols);

/// complex to matrix assignment
void
assign_scalar(std::complex<float> const &value,
	      std::complex<float> *out, stride_type stride,
	      length_type rows, length_type cols);

/// complex to split matrix assignment
inline void
assign_scalar(std::complex<float> const &value,
	      std::pair<float*,float*> out, stride_type stride,
	      length_type rows, length_type cols)
{
  assign_scalar(value.real(), out.first, stride, rows, cols);
  assign_scalar(value.imag(), out.second, stride, rows, cols);
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif

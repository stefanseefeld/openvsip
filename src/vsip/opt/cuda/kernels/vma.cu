/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA kernel for vector multiply-add.

#include <cuComplex.h>
#include "util.hpp"
#include <vsip/support.hpp>
#include <complex>

#include "cmplx.cuh"

using namespace dev;

__global__ void 
k_ma_ss(float const* in1, float const* in2, float const* in3, float* out, size_t length)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;

  int const idx = __mul24(blockDim.x, bx) + tx;
  if (idx < length)
  {
    out[idx] = (in1[idx] * in2[idx]) + in3[idx];
  }
}

__global__ void 
k_ma_cc(cuComplex const* in1, cuComplex const* in2, cuComplex const* in3, cuComplex* out, size_t length)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;

  int const idx = __mul24(blockDim.x, bx) + tx;
  if (idx < length)
  {
    cuComplex tmp;
    cmul(tmp, in1[idx], in2[idx]);
    out[idx].x = tmp.x + in3[idx].x;
    out[idx].y = tmp.y + in3[idx].y;
  }
}



namespace vsip
{
namespace impl
{
namespace cuda
{

void
ma(
  float const*     in1,
  float const*     in2,
  float const*     in3,
  float*           out,
  length_type      length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  k_ma_ss<<<grid, threads>>>(in1, in2, in3, out, length);
}

void
ma(
  std::complex<float> const* in1,
  std::complex<float> const* in2,
  std::complex<float> const* in3,
  std::complex<float>*       out,
  length_type                length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  k_ma_cc<<<grid, threads>>>(reinterpret_cast<cuComplex const *>(in1),
			     reinterpret_cast<cuComplex const *>(in2),
			     reinterpret_cast<cuComplex const *>(in3),
			     reinterpret_cast<cuComplex*>(out),
			     length);
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

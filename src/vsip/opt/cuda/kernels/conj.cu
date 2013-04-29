/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA kernel for complex conjugate.

#include <cuComplex.h>
#include "util.hpp"
#include <vsip/support.hpp>
#include <complex>

#include "cmplx.cuh"

namespace dev
{
// 1-D unit stride
__global__ void
conj(cuComplex const *in, cuComplex *out, size_t length)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;

  int const tid = __mul24(blockDim.x, bx) + tx;
  if (tid < length)
    cconj(out[tid], in[tid]);
}

// 1-D general stride
__global__ void
conj(cuComplex const *in, ptrdiff_t in_stride, cuComplex *out, ptrdiff_t out_stride, size_t length)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;

  int const tid = __mul24(blockDim.x, bx) + tx;
  int const in_idx = __mul24(tid, in_stride);
  int const out_idx = __mul24(tid, out_stride);
  if (tid < length)
    cconj(out[out_idx], in[in_idx]);
}

// 2-D general stride
__global__ void
conj(
  cuComplex const* in,
  ptrdiff_t        row_in_stride,
  ptrdiff_t        col_in_stride,
  cuComplex*       out,
  ptrdiff_t        row_out_stride,
  ptrdiff_t        col_out_stride,
  size_t           num_rows,
  size_t           num_cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;

  int const tidx = __mul24(blockDim.x, bx) + tx;
  int const tidy = __mul24(blockDim.y, by) + ty;
  int const in_idx = __mul24(tidy, row_in_stride) + __mul24(tidx, col_in_stride);
  int const out_idx = __mul24(tidy, row_out_stride) + __mul24(tidx, col_out_stride);
  if (tidy < num_rows && tidx < num_cols)
    cconj(out[out_idx], in[in_idx]);
}
}// namespace dev

namespace vsip
{
namespace impl
{
namespace cuda
{
void
conj(
  std::complex<float> const*    in,
  std::complex<float>*          out,
  length_type                   length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  dev::conj<<<grid, threads>>>(reinterpret_cast<cuComplex const*>(in),
                               reinterpret_cast<cuComplex*>(out), length);
}

void
conj(
  std::complex<float> const*    in,
  stride_type                   in_stride,
  std::complex<float>*          out,
  stride_type                   out_stride,
  length_type                   length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  dev::conj<<<grid, threads>>>(reinterpret_cast<cuComplex const*>(in), in_stride,
                               reinterpret_cast<cuComplex*>(out), out_stride,
                               length);
}

void
conj(
  std::complex<float> const*    in,
  stride_type                   row_in_stride,
  stride_type                   col_in_stride,
  std::complex<float>*          out,
  stride_type                   row_out_stride,
  stride_type                   col_out_stride,
  length_type                   num_rows,
  length_type                   num_cols)
{
  dim3 grid, threads;
  distribute_matrix(num_rows, num_cols, grid, threads);

  dev::conj<<<grid, threads>>>(reinterpret_cast<cuComplex const*>(in),
                               row_in_stride, col_in_stride,
                               reinterpret_cast<cuComplex*>(out),
                               row_out_stride, col_out_stride,
                               num_rows, num_cols);
}
}// namespace vsip::impl::cuda
}// namespace vsip::impl
}// namespace vsip

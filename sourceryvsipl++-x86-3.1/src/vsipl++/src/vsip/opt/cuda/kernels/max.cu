

/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA kernel for maximum.

#include <cuComplex.h>
#include "util.hpp"
#include <vsip/support.hpp>
#include <complex>

namespace dev
{
// 1-D unit stride
__global__ void 
max(float const* in1, float const* in2, float* out, size_t length)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;

  int const idx = __mul24(blockDim.x, bx) + tx;
  if (idx < length)
    out[idx] = fmaxf(in1[idx], in2[idx]);
}
// 1-D general stride
__global__ void 
max(float const* in1, ptrdiff_t in1_stride,
    float const* in2, ptrdiff_t in2_stride,
    float* out, ptrdiff_t out_stride, size_t length)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;

  int const tid = __mul24(blockDim.x, bx) + tx;
  int const in1_idx = __mul24(tid, in1_stride);
  int const in2_idx = __mul24(tid, in2_stride);
  int const out_idx = __mul24(tid, out_stride);
  if (tid < length)
    out[out_idx] = fmaxf(in1[in1_idx], in2[in2_idx]);
}
// 2-D general stride
__global__ void 
max(float const* in1, ptrdiff_t in1_row_stride, ptrdiff_t in1_col_stride,
    float const* in2, ptrdiff_t in2_row_stride, ptrdiff_t in2_col_stride,
    float* out, ptrdiff_t out_row_stride, ptrdiff_t out_col_stride,
    size_t num_rows, size_t num_cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;

  int const tidx = __mul24(blockDim.x, bx) + tx;
  int const tidy = __mul24(blockDim.y, by) + ty;
  int const in1_idx = __mul24(tidy, in1_row_stride) + __mul24(tidx, in1_col_stride);
  int const in2_idx = __mul24(tidy, in2_row_stride) + __mul24(tidx, in2_col_stride);
  int const out_idx = __mul24(tidy, out_row_stride) + __mul24(tidx, out_col_stride);
  if (tidy < num_rows && tidx < num_cols)
    out[out_idx] = fmaxf(in1[in1_idx], in2[in2_idx]);
}
}// namespace dev

namespace vsip
{
namespace impl
{
namespace cuda
{
// 1-D unit stride
void
max(
  float const*     in1,
  float const*     in2,
  float*           out,
  length_type      length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  dev::max<<<grid, threads>>>(in1, in2, out, length);
}
// 1-D general stride
void
max(
  float const*     in1,
  stride_type      in1_stride,
  float const*     in2,
  stride_type      in2_stride,
  float*           out,
  stride_type      out_stride,
  length_type      length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  dev::max<<<grid, threads>>>(in1, in1_stride,
                              in2, in2_stride,
                              out, out_stride, length);
}
// 2-D general stride
void
max(
  float const*     in1,
  stride_type      in1_row_stride,
  stride_type      in1_col_stride,
  float const*     in2,
  stride_type      in2_row_stride,
  stride_type      in2_col_stride,
  float*           out,
  stride_type      out_row_stride,
  stride_type      out_col_stride,
  length_type      nrows,
  length_type      ncols)
{
  dim3 grid, threads;
  distribute_matrix(nrows, ncols, grid, threads);

  dev::max<<<grid, threads>>>(in1, in1_row_stride, in1_col_stride,
                              in2, in2_row_stride, in2_col_stride,
                              out, out_row_stride, out_col_stride,
                              nrows, ncols);
}
} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

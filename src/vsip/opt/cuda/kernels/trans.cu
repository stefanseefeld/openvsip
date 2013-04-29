/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   Transpose CUDA kernels

#include <cuComplex.h>
#include "util.hpp"
#include "trans_fast.cuh"
#include <vsip/support.hpp>

/***********************************************************************
  Device Kernels -- Each thread computes one element
***********************************************************************/


__global__ void 
k_transpose_s(float const* input, float* output, size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if ((row < rows) && (col < cols))
  {
    int const idx_in = __mul24(col, rows) + row;
    int const idx_out = __mul24(row, cols) + col;
    output[idx_out] = input[idx_in];
  }
}

__global__ void 
k_transpose_ip_s(float *inout, size_t size)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if (row < size && col < size && row < col)
  {
    int const idx1 = __mul24(col, size) + row;
    int const idx2 = __mul24(row, size) + col;
    float tmp = inout[idx1];
    inout[idx1] = inout[idx2];
    inout[idx2] = tmp;
  }
}

__global__ void 
k_transpose_c(cuComplex const* input, cuComplex* output, size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if ((row < rows) && (col < cols))
  {
    int const idx_in = __mul24(col, rows) + row;
    int const idx_out = __mul24(row, cols) + col;
    output[idx_out] = input[idx_in];
  }
}

__global__ void 
k_transpose_ip_c(cuComplex *inout, size_t size)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if (row < size && col < size && row < col)
  {
    int const idx1 = __mul24(col, size) + row;
    int const idx2 = __mul24(row, size) + col;
    cuComplex tmp = inout[idx1];
    inout[idx1] = inout[idx2];
    inout[idx2] = tmp;
  }
}

namespace vsip
{
namespace impl
{
namespace cuda
{

void
transpose(float const *in, float *out,
	  length_type rows, length_type cols)
{
  int const tile_size = VSIP_CUDA_TRANS_BLOCK_DIM;
  if ((rows % tile_size) || (cols % tile_size))
  {
    dim3 blocks, threads;
    distribute_matrix(rows, cols, blocks, threads);

    k_transpose_s<<<blocks, threads>>>(in, out, rows, cols);
  }   
  else
  {
    dim3 blocks(rows/tile_size, cols/tile_size);
    dim3 threads(tile_size, tile_size);

    k_fast_transpose_s<<<blocks, threads>>>(in, out, rows, cols);
  }
}

void
transpose(float *inout, length_type size)
{
  dim3 blocks, threads;
  distribute_matrix(size, size, blocks, threads);
  k_transpose_ip_s<<<blocks, threads>>>(inout, size);
}

void
transpose(std::complex<float> const *in, std::complex<float> *out,
	  length_type rows, length_type cols)
{
  int const tile_size = VSIP_CUDA_TRANS_BLOCK_DIM;
  if ((rows % tile_size) || (cols % tile_size))
  {
    dim3 blocks, threads;
    distribute_matrix(rows, cols, blocks, threads);

    k_transpose_c<<<blocks, threads>>>(reinterpret_cast<cuComplex const*>(in),
                                       reinterpret_cast<cuComplex *>(out),
                                       rows, cols);
  }
  else
  {
    dim3 blocks(rows/tile_size, cols/tile_size);
    dim3 threads(tile_size, tile_size);

    k_fast_transpose_c<<<blocks, threads>>>(reinterpret_cast<cuComplex const*>(in),
                                            reinterpret_cast<cuComplex *>(out),
                                            rows, cols);
  }
}

void
transpose(std::complex<float> *inout, length_type size)
{
  dim3 blocks, threads;
  distribute_matrix(size, size, blocks, threads);
  k_transpose_ip_c<<<blocks, threads>>>
    (reinterpret_cast<cuComplex*>(inout), size);
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

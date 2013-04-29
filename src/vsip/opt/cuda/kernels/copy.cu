/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   Copy CUDA kernels

#include <cuda_runtime.h>
#include <complex>
#include <cuComplex.h>
#include "util.hpp"
#include <vsip/support.hpp>

/***********************************************************************
  Device Kernels -- Each thread computes 'size' elements
***********************************************************************/

#define MAX_SHARED  (16*1024)
#define THREADS  512

__global__ void 
k_vcopy_s(float const* input, float* output, size_t size)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int const idx = __mul24(blockDim.x, bx) + tx;

  if (idx < size)
    output[idx] = input[idx];
}

__global__ void 
k_vcopy_strided_s(float const* input, ptrdiff_t in_stride,
		  float* output, ptrdiff_t out_stride,
		  size_t size)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int const idx = __mul24(blockDim.x, bx) + tx;
  int const in_idx = __mul24(blockDim.x, bx) + __mul24(in_stride, tx);
  int const out_idx = __mul24(blockDim.x, bx) + __mul24(out_stride, tx);

  if (idx < size)
    output[out_idx] = input[in_idx];
}

__global__ void 
k_vcopy_c(cuComplex const *input, cuComplex *output, size_t size)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int const idx = __mul24(blockDim.x, bx) + tx;

  if (idx < size)
    output[idx] = input[idx];
}

__global__ void 
k_vcopy_strided_c(cuComplex const *input, ptrdiff_t in_stride,
		  cuComplex *output, ptrdiff_t out_stride,
		  size_t size)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int const idx = __mul24(blockDim.x, bx) + tx;
  int const in_idx = __mul24(blockDim.x, bx) + __mul24(in_stride, tx);
  int const out_idx = __mul24(blockDim.x, bx) + __mul24(out_stride, tx);

  if (idx < size)
    output[out_idx] = input[in_idx];
}

__global__ void 
k_mcopy_s(float const *input, ptrdiff_t in_stride,
	  float *output, ptrdiff_t out_stride,
	  size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if ((row < rows) && (col < cols))
  {
    int const in_idx = __mul24(row, in_stride) + col;
    int const out_idx = __mul24(row, out_stride) + col;
    output[out_idx] = input[in_idx];
  }
}

__global__ void 
k_mcopy_strided_s(float const *input, ptrdiff_t in_r_stride, ptrdiff_t in_c_stride,
		  float *output, ptrdiff_t out_r_stride, ptrdiff_t out_c_stride,
		  size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if ((row < rows) && (col < cols))
  {
    int const in_idx = __mul24(row, in_r_stride) + __mul24(col, in_c_stride);
    int const out_idx = __mul24(row, out_r_stride) + __mul24(col, out_c_stride);
    output[out_idx] = input[in_idx];
  }
}

__global__ void 
k_mcopy_c(cuComplex const *input, ptrdiff_t in_stride,
	  cuComplex *output, ptrdiff_t out_stride,
	  size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if ((row < rows) && (col < cols))
  {
    int const in_idx = __mul24(row, in_stride) + col;
    int const out_idx = __mul24(row, out_stride) + col;
    output[out_idx] = input[in_idx];
  }
}

__global__ void 
k_mcopy_strided_c(cuComplex const *input, ptrdiff_t in_r_stride, ptrdiff_t in_c_stride,
		  cuComplex *output, ptrdiff_t out_r_stride, ptrdiff_t out_c_stride,
		  size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if ((row < rows) && (col < cols))
  {
    int const in_idx = __mul24(row, in_r_stride) + __mul24(col, in_c_stride);
    int const out_idx = __mul24(row, out_r_stride) + __mul24(col, out_c_stride);
    output[out_idx] = input[in_idx];
  }
}

// size must be less than or equal to MAX_SHARED/sizeof(float)
__global__ void
dev2shared(float* dev, size_t size)
{
  // shared memory
  // the size is determined by the host application
  extern  __shared__  float sdata[];

  dev += size * threadIdx.x;
  float* dst = sdata + size * threadIdx.x;

  for (size_t i = 0; i < size; ++i)
    dst[i] = dev[i];
  __syncthreads();
}

__global__ void 
k_assign_scalar_s(float s, float *output, size_t size)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int const idx = __mul24(blockDim.x, bx) + tx;

  if (idx < size)
    output[idx] = s;
}

__global__ void 
k_assign_scalar_c(cuComplex s, cuComplex *output, size_t size)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int const idx = __mul24(blockDim.x, bx) + tx;

  if (idx < size)
    output[idx] = s;
}

__global__ void 
k_massign_scalar_s(float s, float *output, ptrdiff_t stride,
		   size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if (row < rows && col < cols)
  {
    int const idx = __mul24(row, stride) + col;
    output[idx] = s;
  }
}

__global__ void 
k_massign_scalar_c(cuComplex s, cuComplex *output, ptrdiff_t stride,
		   size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if (row < rows && col < cols)
  {
    int const idx = __mul24(row, stride) + col;
    output[idx] = s;
  }
}

namespace vsip
{
namespace impl
{
namespace cuda
{

/// dense scalar vector copy
void
copy(float const *in, float *out, length_type length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  k_vcopy_s<<<grid, threads>>>(in, out, length);
}

/// dense complex vector copy
void
copy(std::complex<float> const *in,
     std::complex<float> *out, length_type length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  k_vcopy_c<<<grid, threads>>>
    (reinterpret_cast<cuComplex const *>(in),
     reinterpret_cast<cuComplex *>(out), length);
}

/// scalar vector copy
void
copy(float const *in, stride_type in_stride,
     float *out, stride_type out_stride,
     length_type length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  k_vcopy_strided_s<<<grid, threads>>>(in, in_stride, out, out_stride, length);
}

/// complex vector copy
void
copy(std::complex<float> const *in, stride_type in_stride,
     std::complex<float> *out, stride_type out_stride, length_type length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  k_vcopy_strided_c<<<grid, threads>>>
    (reinterpret_cast<cuComplex const *>(in), in_stride,
     reinterpret_cast<cuComplex *>(out), out_stride, length);
}

/// dense float matrix copy
void
copy(float const *in, stride_type in_stride,
     float *out, stride_type out_stride,
     length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  k_mcopy_s<<<blocks, threads>>>(in, in_stride, out, out_stride, rows, cols);
}

/// dense complex matrix copy
void
copy(std::complex<float> const *in, stride_type in_stride,
     std::complex<float> *out, stride_type out_stride,
     length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  k_mcopy_c<<<blocks, threads>>>
    (reinterpret_cast<cuComplex const *>(in), in_stride,
     reinterpret_cast<cuComplex *>(out), out_stride,
     rows, cols);
}

/// float matrix copy
void
copy(float const *in, stride_type in_stride_0, stride_type in_stride_1,
     float *out, stride_type out_stride_0, stride_type out_stride_1,
     length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  k_mcopy_strided_s<<<blocks, threads>>>(in, in_stride_0, in_stride_1,
					 out, out_stride_0, out_stride_1,
					 rows, cols);
}

/// complex matrix copy
void
copy(std::complex<float> const *in, stride_type in_stride_0, stride_type in_stride_1,
     std::complex<float> *out, stride_type out_stride_0, stride_type out_stride_1,
     length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  k_mcopy_strided_c<<<blocks, threads>>>
    (reinterpret_cast<cuComplex const *>(in), in_stride_0, in_stride_1,
     reinterpret_cast<cuComplex *>(out), out_stride_0, out_stride_1,
     rows, cols);
}

void
assign_scalar(float value, float *out, length_type length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  k_assign_scalar_s<<<grid, threads>>>(value, out, length);
}

void
assign_scalar(std::complex<float> const &value,
	      std::complex<float> *out, length_type length)
{
  dim3 grid, threads;
  distribute_vector(length, grid, threads);

  k_assign_scalar_c<<<grid, threads>>>
    (*reinterpret_cast<cuComplex const *>(&value),
     reinterpret_cast<cuComplex *>(out), length);
}

void
assign_scalar(float value, float *out, stride_type stride,
	      length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  k_massign_scalar_s<<<blocks, threads>>>(value, out, stride, rows, cols);
}

void
assign_scalar(std::complex<float> const &value,
	      std::complex<float> *out, stride_type stride,
	      length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  k_massign_scalar_c<<<blocks, threads>>>
    (*reinterpret_cast<cuComplex const *>(&value),
     reinterpret_cast<cuComplex *>(out), stride, rows, cols);
}

/// FIXME: Who needs this ?
void
copy_device_to_shared(float* src, size_t size)
{
  // When the size exceeds the maximum amount of shared memory,
  // break the transfer up into pieces.  In a realistic application
  // one would never exceed the maximum like this, but this is for 
  // performance testing purposes only.
  while (size) 
  {
    int count = (size > MAX_SHARED) ? MAX_SHARED : size;

    dev2shared<<<THREADS, 1, MAX_SHARED>>>(src, count/THREADS);

    src += count;
    size -= count;
  }
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

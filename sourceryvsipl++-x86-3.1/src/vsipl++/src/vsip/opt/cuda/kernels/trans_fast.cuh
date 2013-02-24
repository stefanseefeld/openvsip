/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/** @file    trans_fast.cuh
    @author  Don McCoy
    @date    2010-03-10
    @brief   VSIPL++ Library: Fast matrix transpose for CUDA

    This software contains source code provided by NVIDIA Corporation.
*/

#ifndef VSIP_OPT_CUDA_KERNELS_TRANS_FAST_HPP
#define VSIP_OPT_CUDA_KERNELS_TRANS_FAST_HPP


#define VSIP_CUDA_TRANS_BLOCK_DIM 16

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than a naive transpose kernel.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.


// Size parameters refer to output matrix dimensions
__global__ void 
k_fast_transpose_s(float const* input, float* output, size_t rows, size_t cols)
{
  __shared__ float tile[VSIP_CUDA_TRANS_BLOCK_DIM][VSIP_CUDA_TRANS_BLOCK_DIM+1];

  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x * VSIP_CUDA_TRANS_BLOCK_DIM;
  int const by = blockIdx.y * VSIP_CUDA_TRANS_BLOCK_DIM;
  int const col = by + tx;
  int const row = bx + ty;

  if ((row < rows) && (col < cols))
  {
    int const idx_in = __mul24((by + ty), rows) + bx + tx;
    tile[ty][tx] = input[idx_in];
  }

  __syncthreads();

  if ((row < rows) && (col < cols))
  {
    int const idx_out = __mul24(row, cols) + col;
    output[idx_out] = tile[tx][ty];
  }
}


// Size parameters refer to output matrix dimensions
__global__ void 
k_fast_transpose_c(cuComplex const* input, cuComplex* output, size_t rows, size_t cols)
{
  __shared__ cuComplex tile[VSIP_CUDA_TRANS_BLOCK_DIM][VSIP_CUDA_TRANS_BLOCK_DIM+1];

  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x * VSIP_CUDA_TRANS_BLOCK_DIM;
  int const by = blockIdx.y * VSIP_CUDA_TRANS_BLOCK_DIM;
  int const col = by + tx;
  int const row = bx + ty;

  if ((row < rows) && (col < cols))
  {
    int const idx_in = __mul24((by + ty), rows) + bx + tx;
    tile[ty][tx] = input[idx_in];
  }

  __syncthreads();

  if ((row < rows) && (col < cols))
  {
    int const idx_out = __mul24(row, cols) + col;
    output[idx_out] = tile[tx][ty];
  }
}

#endif  // VSIP_OPT_CUDA_KERNELS_TRANS_FAST_HPP
/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/** @file    trans_fast.cuh
    @author  Don McCoy
    @date    2010-03-10
    @brief   VSIPL++ Library: Fast matrix transpose for CUDA

    This software contains source code provided by NVIDIA Corporation.
*/

#ifndef VSIP_OPT_CUDA_KERNELS_TRANS_FAST_HPP
#define VSIP_OPT_CUDA_KERNELS_TRANS_FAST_HPP


/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#define VSIP_CUDA_TRANS_BLOCK_DIM 16

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.


// Size parameters refer to output matrix dimensions
__global__ void 
k_fast_transpose_s(float const* input, float* output, size_t rows, size_t cols)
{
  __shared__ float tile[VSIP_CUDA_TRANS_BLOCK_DIM][VSIP_CUDA_TRANS_BLOCK_DIM+1];

  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x * VSIP_CUDA_TRANS_BLOCK_DIM;
  int const by = blockIdx.y * VSIP_CUDA_TRANS_BLOCK_DIM;
  int const col = by + tx;
  int const row = bx + ty;

  if ((row < rows) && (col < cols))
  {
    int const idx_in = __mul24((by + ty), rows) + bx + tx;
    tile[ty][tx] = input[idx_in];
  }

  __syncthreads();

  if ((row < rows) && (col < cols))
  {
    int const idx_out = __mul24(row, cols) + col;
    output[idx_out] = tile[tx][ty];
  }
}


// Size parameters refer to output matrix dimensions
__global__ void 
k_fast_transpose_c(cuComplex const* input, cuComplex* output, size_t rows, size_t cols)
{
  __shared__ cuComplex tile[VSIP_CUDA_TRANS_BLOCK_DIM][VSIP_CUDA_TRANS_BLOCK_DIM+1];

  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x * VSIP_CUDA_TRANS_BLOCK_DIM;
  int const by = blockIdx.y * VSIP_CUDA_TRANS_BLOCK_DIM;
  int const col = by + tx;
  int const row = bx + ty;

  if ((row < rows) && (col < cols))
  {
    int const idx_in = __mul24((by + ty), rows) + bx + tx;
    tile[ty][tx] = input[idx_in];
  }

  __syncthreads();

  if ((row < rows) && (col < cols))
  {
    int const idx_out = __mul24(row, cols) + col;
    output[idx_out] = tile[tx][ty];
  }
}

#endif  // VSIP_OPT_CUDA_KERNELS_TRANS_FAST_HPP

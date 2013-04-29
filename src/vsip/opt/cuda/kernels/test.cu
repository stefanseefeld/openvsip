/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    test.cu
    @author  Don McCoy
    @date    2009-06-21
    @brief   VSIPL++ Library: CUDA kernels used soley for testing.
               May also be used as a template for new kernels.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cuComplex.h>

#include "util.hpp"


/***********************************************************************
  Device Kernels -- Each thread computes one element
***********************************************************************/

__global__ void 
k_null_s(float* inout, size_t rows, size_t cols)
{
}

__global__ void 
k_null_c(cuComplex* inout, size_t rows, size_t cols)
{
}


__global__ void 
k_check_distrib(size_t* inout, size_t size)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int const idx = __mul24(blockDim.x, bx) + tx;

  // Threads outside the bounds of the vector do no work.  This makes
  // it possible to always divide the thread blocks into efficiently-
  // sized pieces.
  //
  if (idx < size)
  { 
    // Write the linear index into each element of the matrix
    inout[idx] = (size_t)idx;
  }
}

__global__ void 
k_check_distrib(size_t* inout, size_t rows, size_t cols)
{
  // Using the two-layer coordinate system (grid + thread block),
  // determine which row and column of the matrix to compute.
  int const ty = threadIdx.y;
  int const tx = threadIdx.x;
  int const by = blockIdx.y;
  int const bx = blockIdx.x;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  // Threads outside the bounds of the matrix do no work.  This makes
  // it possible to always divide the thread blocks into efficiently-
  // sized pieces.
  //
  if ((row < rows) && (col < cols))
  { 
    // Write the linear index into each element of the matrix
    int const idx = __mul24(col, rows) + row;
    inout[idx] = (size_t)idx;
  }
}



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cuda
{

void
null_s(
  float* inout,
  size_t rows,
  size_t cols)
{
  dim3 grid, threads;
  distribute_matrix(rows, cols, grid, threads);

  k_null_s<<<grid, threads>>>(inout, rows, cols);
  cudaThreadSynchronize();
}

void
null_c(
  cuComplex* inout,
  size_t     rows,
  size_t     cols)
{
  dim3 grid, threads;
  distribute_matrix(rows, cols, grid, threads);

  k_null_c<<<grid, threads>>>(inout, rows, cols);
  cudaThreadSynchronize();
}


void
check_distrib(
  size_t* inout,
  size_t  size)
{
  dim3 grid, threads;
  distribute_vector(size, grid, threads);

  k_check_distrib<<<grid, threads>>>(inout, size);
  cudaThreadSynchronize();
}

void
check_distrib(
  size_t* inout,
  size_t rows,
  size_t cols)
{
  dim3 grid, threads;
  distribute_matrix(rows, cols, grid, threads);

  k_check_distrib<<<grid, threads>>>(inout, rows, cols);
  cudaThreadSynchronize();
}


} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

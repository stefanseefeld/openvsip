/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    fftshift.cu
    @author  Don McCoy
    @date    2009-07-02
    @brief   VSIPL++ Library: CUDA Kernel for matrix fftshift
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
k_fftshift_s(float const* input, float* output, size_t rows, size_t cols, 
		   unsigned int in_major_dim, unsigned int out_major_dim)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  int const r2 = rows / 2;
  int const c2 = cols / 2;
  if ((row < rows) && (col < cols))
  { 
    int swap_row = (row < r2) ? row + r2  : row - r2;
    int swap_col = (col < c2) ? col + c2  : col - c2;
      
    int const idx_in = (in_major_dim == 0) ? 
      __mul24(swap_row, cols) + swap_col :
      __mul24(swap_col, rows) + swap_row;
    int const idx_out = (out_major_dim == 0) ? 
      __mul24(row, cols) + col :
      __mul24(col, rows) + row;
    output[idx_out] = input[idx_in];
  }
}

__global__ void 
k_fftshift_c(cuComplex const* input, cuComplex* output, size_t rows, size_t cols, 
		       unsigned int in_major_dim, unsigned int out_major_dim)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  int const r2 = rows / 2;
  int const c2 = cols / 2;
  if ((row < rows) && (col < cols))
  { 
    int swap_row = (row < r2) ? row + r2  : row - r2;
    int swap_col = (col < c2) ? col + c2  : col - c2;
      
    int const idx_in = (in_major_dim == 0) ? 
      __mul24(swap_row, cols) + swap_col :
      __mul24(swap_col, rows) + swap_row;
    int const idx_out = (out_major_dim == 0) ? 
      __mul24(row, cols) + col :
      __mul24(col, rows) + row;
    output[idx_out] = input[idx_in];
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
fftshift_s(
  float const* input,  
  float*       output,
  size_t       rows,
  size_t       cols,
  unsigned int in_major_dim, 
  unsigned int out_major_dim)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  k_fftshift_s<<<blocks, threads>>>(input, output, rows, cols, in_major_dim, out_major_dim);
}


void
fftshift_c(
  cuComplex const* input,
  cuComplex*       output,
  size_t           rows,
  size_t           cols,
  unsigned int     in_major_dim, 
  unsigned int     out_major_dim)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  k_fftshift_c<<<blocks, threads>>>(input, output, rows, cols, in_major_dim, out_major_dim);
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

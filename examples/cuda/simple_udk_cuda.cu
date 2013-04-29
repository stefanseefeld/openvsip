#include <stdio.h>
/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   Perform a simple copy

#include <cuda_runtime.h>

/// Each CUDA thread runs an instance of this function, which copies 
/// a single element.
__global__ void 
copy_kernel(float const *input, ptrdiff_t in_r_stride, ptrdiff_t in_c_stride,
            float *output, ptrdiff_t out_r_stride, ptrdiff_t out_c_stride,
            size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  // When there are more threads than elements to be processed, the
  // bounds must be checked first.
  if ((row < rows) && (col < cols))
  {
    int const in_idx = __mul24(row, in_r_stride) + __mul24(col, in_c_stride);
    int const out_idx = __mul24(row, out_r_stride) + __mul24(col, out_c_stride);
    output[out_idx] = input[in_idx];
  }
}

/// Calculates the optimal grid and thread-block sizes for a simple
/// distribution of a matrix.  At least one thread for every element 
/// is created, with the actual number rounded up in each dimension
/// to create fully populated blocks.  Excess threads must be kept
/// idle by ensuring they are within bounds of the actual matrix.
///
///   :grid:  set to the minimum number of thread blocks
///           needed to accomodate a matrix of the given size
///
///   :threads:  set to the size of the thread block
///
inline 
void
distribute_matrix(size_t rows, size_t cols, dim3& grid, dim3& threads)
{
  // Optimal thread blocks are as large as possible in order to 
  // maximize occupancy.  There is a maximum of 512 threads per block
  // allowed.
  threads.y = 16; // rows
  threads.x = 32; // cols

  grid.y = (rows + threads.y - 1) / threads.y;
  grid.x = (cols + threads.x - 1) / threads.x;
}

/// This is the CUDA kernel entry point.
/// This function is executed on the host. It spawns device threads to perform
/// the actual operation in device memory.
void
udk_copy(float const *input, ptrdiff_t in_r_stride, ptrdiff_t in_c_stride,
         float *output, ptrdiff_t out_r_stride, ptrdiff_t out_c_stride,
         size_t rows, size_t cols)
{
  dim3 threads, blocks;
  distribute_matrix(rows, cols, blocks, threads);

  copy_kernel<<<blocks, threads>>>(input, in_r_stride, in_c_stride,
                                   output, out_r_stride, out_c_stride,
                                   rows, cols);
}

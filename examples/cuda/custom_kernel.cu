/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   Sourcery VSIPL++ example demonstrating how to write a custom
///   CUDA kernel that interoperates with SV++ views.


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


/// This is the CUDA kernel entry point executed on the host. It spawns the
/// device threads that perform the actual operation in device (global) memory.
void
copy(float const *input, ptrdiff_t in_r_stride, ptrdiff_t in_c_stride,
     float *output, ptrdiff_t out_r_stride, ptrdiff_t out_c_stride,
     size_t rows, size_t cols)
{
  dim3 threads, grid;
  threads.y = 16; // rows
  threads.x = 32; // cols
  grid.y = (rows + threads.y - 1) / threads.y;
  grid.x = (cols + threads.x - 1) / threads.x;

  copy_kernel<<<grid, threads>>>(
    input, in_r_stride, in_c_stride,
    output, out_r_stride, out_c_stride,
    rows, cols);
}

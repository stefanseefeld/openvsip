/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA Kernel for Vector-Matrix Multiplication

#include <cuComplex.h>
#include "util.hpp"
#include <vsip/support.hpp>
#include <complex>


/***********************************************************************
  Device functions (callable only via kernels)
***********************************************************************/

#include "cmplx.cuh"

namespace dev
{
/***********************************************************************
  Device Kernels -- Each thread computes one element
***********************************************************************/

// Support provided for some (but not all) combinations of real and complex, 
// single and double precision.  The BLAS notation convention is used to 
// indicate the types of the two inputs and the result, i.e.:
//
//   S = single precision real      D = double precision real
//   C = single precision complex   Z = double precision complex


// Scalar-Matrix multiply         s * S --> S
__global__ void 
k_smmul(float const scale, float const* input, float* output, 
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
    int const idx = __mul24(row, cols) + col;
    output[idx] = scale * input[idx];
  }
}

// Scalar-Matrix multiply         s * C --> C
__global__ void 
k_smmul(float const scale, cuComplex const* input, cuComplex* output, 
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
    int const idx = __mul24(row, cols) + col;
    scmul(output[idx], scale, input[idx]);
  }
}


// Vector-Matrix multiply         S * S --> S
__global__ void 
vmmul(float const* kernel, float const* input, float* output, 
          int rows, int cols, bool by_row)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const tidy = __mul24(blockDim.y, by) + ty;
  int const tidx = __mul24(blockDim.x, bx) + tx;
  int x_max, y_max, idx;
    
  if (by_row)
  {
    x_max = cols;
    y_max = rows;
    idx = __mul24(tidy, cols) + tidx;
  }
  else
  {
    x_max = rows;
    y_max = cols;
    idx = __mul24(tidx, cols) + tidy;
  }

  //  Use 32 elements since the block is created with a maximum of 32 threads
  //   in either dimension
  __shared__ float kernel_sh[32];

  if (tidx >= x_max || tidy >= y_max)
    return;

  if (ty == 0)
    kernel_sh[tx] = kernel[tidx];

  __syncthreads();

  output[idx] = __fmul_rn(kernel_sh[tx], input[idx]);
}

// Vector-Matrix multiply         S * S --> S
//   Kernel for large number of rows when performing a "by column" vector-matrix
//   multiply.
__global__ void
vmmul_by_col_long(float const* kernel, float const* input, float* output, 
         int rows, int cols, int nloops)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int bidx = __mul24(bx, cols);

  // Each thread only uses a single element of the kernel since an independent
  //   block is created for each row and the kernel aligns a single element with
  //   each row.
  __shared__ float kernel_sh;

  if (tx == 0)
    kernel_sh = kernel[bx];

  __syncthreads();

  // Unroll loops 16 times, this number was determined via trial and error for
  //  performance.
  #pragma unroll 16
  for (int i = 0; i < nloops; ++i)
  {
    // Each block is created with 64 threads, thus each iteration computes
    //  64 elements.  
    int id = __mul24(i, 64) + tx;
    if (id < cols)
    {
      int idx = bidx + id;
      output[idx] = __fmul_rn(kernel_sh, input[idx]);
    }
  }
}

// Vector-Matrix multiply         S * C --> C
__global__ void 
vmmul(float const* kernel, cuComplex const* input, cuComplex* output, 
         int rows, int cols, bool by_row)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const tidy = __mul24(blockDim.y, by) + ty;
  int const tidx = __mul24(blockDim.x, bx) + tx;
  int x_max, y_max, idx;
    
  if (by_row)
  {
    x_max = cols;
    y_max = rows;
    idx = __mul24(tidy, cols) + tidx;
  }
  else
  {
    x_max = rows;
    y_max = cols;
    idx = __mul24(tidx, cols) + tidy;
  }

  //  Use 32 elements since the block is created with a maximum of 32 threads
  //   in either dimension
  __shared__ float kernel_sh[32];

  if (tidx >= x_max || tidy >= y_max)
    return;

  if (ty == 0)
    kernel_sh[tx] = kernel[tidx];

  __syncthreads();

  scmul(output[idx], kernel_sh[tx], input[idx]);
}

// Vector-Matrix multiply         S * C --> C
//   Kernel for large number of rows when performing a "by column" vector-matrix
//   multiply.
__global__ void
vmmul_by_col_long(float const* kernel, cuComplex const* input, cuComplex* output, 
         int rows, int cols, int nloops)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int bidx = __mul24(bx, cols);

  // Each thread only uses a single element of the kernel since an independent
  //   block is created for each row and the kernel aligns a single element with
  //   each row.
  __shared__ float kernel_sh;

  if (tx == 0)
    kernel_sh = kernel[bx];

  __syncthreads();

  // Unroll loops 16 times, this number was determined via trial and error for
  //  performance.
  #pragma unroll 16
  for (int i = 0; i < nloops; ++i)
  {
    // Each block is created with 64 threads, thus each iteration computes
    //  64 elements.  
    int id = __mul24(i, 64) + tx;
    if (id < cols)
    {
      int idx = bidx + id;
      scmul(output[idx], kernel_sh, input[idx]);
    }
  }
}

// Vector-Matrix multiply         C * S --> C
__global__ void 
vmmul(cuComplex const* kernel, float const* input, cuComplex* output, 
         int rows, int cols, bool by_row)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const tidy = __mul24(blockDim.y, by) + ty;
  int const tidx = __mul24(blockDim.x, bx) + tx;
  int x_max, y_max, idx;
  
  if (by_row)
  {
    x_max = cols;
    y_max = rows;
    idx = __mul24(tidy, cols) + tidx;
  }
  else
  {
    x_max = rows;
    y_max = cols;
    idx = __mul24(tidx, cols) + tidy;
  }

  //  Use 32 elements since the block is created with a maximum of 32 threads
  //   in either dimension
  __shared__ cuComplex kernel_sh[32];

  if (tidx >= x_max || tidy >= y_max)
    return;

  if (ty == 0)
    kernel_sh[tx] = kernel[tidx];

  __syncthreads();

  scmul(output[idx], input[idx], kernel_sh[tx]);
}

// Vector-Matrix multiply         C * S --> C
//   Kernel for large number of rows when performing a "by column" vector-matrix
//   multiply.
__global__ void
vmmul_by_col_long(cuComplex const* kernel, float const* input, cuComplex* output, 
         int rows, int cols, int nloops)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int bidx = __mul24(bx, cols);

  // Each thread only uses a single element of the kernel since an independent
  //   block is created for each row and the kernel aligns a single element with
  //   each row.
  __shared__ cuComplex kernel_sh;

  if (tx == 0)
    kernel_sh = kernel[bx];

  __syncthreads();

  // Unroll loops 16 times, this number was determined via trial and error for
  //  performance.
  #pragma unroll 16
  for (int i = 0; i < nloops; ++i)
  {
    // Each block is created with 64 threads, thus each iteration computes
    //  64 elements.  
    int id = __mul24(i, 64) + tx;
    if (id < cols)
    {
      int idx = bidx + id;
      scmul(output[idx], input[idx], kernel_sh);
    }
  }
}

// Vector-Matrix multiply         C * C --> C
__global__ void 
vmmul(cuComplex const* kernel, cuComplex const* input, cuComplex* output, 
         int rows, int cols, bool by_row)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const tidy = __mul24(blockDim.y, by) + ty;
  int const tidx = __mul24(blockDim.x, bx) + tx;
  int x_max, y_max, idx;

  if (by_row)
  {
    x_max = cols;
    y_max = rows;
    idx = __mul24(tidy, cols) + tidx;
  }
  else
  {
    x_max = rows;
    y_max = cols;
    idx = __mul24(tidx, cols) + tidy;
  }

  //  Use 32 elements since the block is created with a maximum of 32 threads
  //   in either dimension
  __shared__ cuComplex kernel_sh[32];

  if (tidx >= x_max || tidy >= y_max)
    return;

  if (ty == 0)
    kernel_sh[tx] = kernel[tidx];

  __syncthreads();

  cmul(output[idx], kernel_sh[tx], input[idx]);
}

// Vector-Matrix multiply         C * C --> C
//   Kernel for large number of rows when performing a "by column" vector-matrix
//   multiply.
__global__ void
vmmul_by_col_long(cuComplex const* kernel, cuComplex const* input, cuComplex* output, 
         int rows, int cols, int nloops)
{
  int const tx = threadIdx.x;
  int const bx = blockIdx.x;
  int bidx = __mul24(bx, cols);

  // Each thread only uses a single element of the kernel since an independent
  //   block is created for each row and the kernel aligns a single element with
  //   each row.
  __shared__ cuComplex kernel_sh;

  if (tx == 0)
    kernel_sh = kernel[bx];

  __syncthreads();

  // Unroll loops 16 times, this number was determined via trial and error for
  //  performance.
  #pragma unroll 16
  for (int i = 0; i < nloops; ++i)
  {
    // Each block is created with 64 threads, thus each iteration computes
    //  64 elements.  
    int id = __mul24(i, 64) + tx;
    if (id < cols)
    {
      int idx = bidx + id;
      cmul(output[idx], kernel_sh, input[idx]);
    }
  }
}

// Vector-Matrix multiply with scale		C * C * s --> C
//   Computes A = A * B * c  where A is a complex matrix, B is a complex 
//   vector and c is real.  This is used to combine the scaling step from 
//   an inverse FFT with the vector-multiplication step when doing fast 
//   convolution.
__global__ void 
vmmuls(cuComplex const* kernel, cuComplex const* input, cuComplex* output, 
          float scale, size_t rows, size_t cols, bool by_row)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;
  int const vec_idx = by_row ? col : row;

  if ((row < rows) && (col < cols))
  {
    int const idx = __mul24(row, cols) + col;
    cmuls(output[idx], kernel[vec_idx], input[idx], scale);
  }
}


// Matrix-Matrix multiply with scale		C * C * s --> C
//   Computes A = A * B * c  where A is a complex matrix, B is a complex 
//   vector and c is real.  This is used to combine the scaling step from 
//   an inverse FFT with the vector-multiplication step when doing fast 
//   convolution.
__global__ void 
matmuls(cuComplex const* kernel, cuComplex const* input, cuComplex* output, 
           float scale, size_t rows, size_t cols)
{
  int const tx = threadIdx.x;
  int const ty = threadIdx.y;
  int const bx = blockIdx.x;
  int const by = blockIdx.y;
  int const row = __mul24(blockDim.y, by) + ty;
  int const col = __mul24(blockDim.x, bx) + tx;

  if ((row < rows) && (col < cols))
  {
    int const idx = __mul24(row, cols) + col;
    cmuls(output[idx], kernel[idx], input[idx], scale);
  }
}
} // namespace dev

namespace vsip
{
namespace impl
{
namespace cuda
{

void
smmul(float const scale, float const *in, float *out,
      length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  dev::k_smmul<<<blocks, threads>>>(scale, in, out, rows, cols);
}

void
smmul(float const scale,
      std::complex<float> const *in,
      std::complex<float> *out,
      length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  dev::k_smmul<<<blocks, threads>>>
    (scale,
     reinterpret_cast<cuComplex const *>(in),
     reinterpret_cast<cuComplex*>(out),
     rows, cols);
}


void 
vmmul_row(float const *kernel, float const *in, float *out,
	  length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  bool by_row = true;
  dev::vmmul<<<blocks, threads>>>(kernel, in, out, rows, cols, by_row);
}

void 
vmmul_row(float const *kernel,
	  std::complex<float> const *in,
	  std::complex<float> *out,
	  length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  bool by_row = true;
  dev::vmmul<<<blocks, threads>>>
    (kernel,
     reinterpret_cast<cuComplex const*>(in),
     reinterpret_cast<cuComplex*>(out), rows, cols, by_row);
}

void 
vmmul_row(std::complex<float> const *kernel,
	  float const *in,
	  std::complex<float> *out,
	  length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  bool by_row = true;
  dev::vmmul<<<blocks, threads>>>
    (reinterpret_cast<cuComplex const*>(kernel),
     in,
     reinterpret_cast<cuComplex *>(out),
     rows, cols, by_row);
}

void 
vmmul_row(std::complex<float> const *kernel,  
	  std::complex<float> const *in,
	  std::complex<float> *out,
	  length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  bool by_row = true;
  dev::vmmul<<<blocks, threads>>>
    (reinterpret_cast<cuComplex const*>(kernel),
     reinterpret_cast<cuComplex const*>(in),
     reinterpret_cast<cuComplex*>(out), rows, cols, by_row);
}


void 
vmmul_col(float const *kernel, float const *in, float *out,
	  length_type rows, length_type cols)
{
  dim3 blocks, threads, blocks_by_col, threads_by_col;
  distribute_matrix(rows, cols, blocks, threads);

  // Create each block with 64 threads and create one block for each row. This
  //  only applies to the "by_col_long" kernel. 
  threads_by_col.x = 64;
  blocks_by_col.x = rows;

  //  Required number of loops for "by_col_long" kernel.
  int num_loops = (cols + threads_by_col.x - 1) / threads_by_col.x;

  bool by_row = false;
  if (rows < 128)
  {
    dev::vmmul<<<blocks, threads>>>(kernel, in, out, rows, cols, by_row);
  }
  else
  {
    dev::vmmul_by_col_long<<<blocks_by_col, threads_by_col>>>(kernel, in, out,
							  rows, cols, num_loops);
  }
}

void 
vmmul_col(float const *kernel,
	  std::complex<float> const *in,
	  std::complex<float> *out,
	  length_type rows, length_type cols)
{
  dim3 blocks, threads, blocks_by_col, threads_by_col;
  distribute_matrix(rows, cols, blocks, threads);

  // Create each block with 64 threads and create one block for each row. This
  //  only applies to the "by_col_long" kernel. 
  threads_by_col.x = 64;
  blocks_by_col.x = rows;

  //  Required number of loops for "by_col_long" kernel.
  int num_loops = (cols + threads_by_col.x - 1) / threads_by_col.x;

  bool by_row = false;
  if (rows < 128)
  {
    dev::vmmul<<<blocks, threads>>>
      (kernel,
       reinterpret_cast<cuComplex const*>(in),
       reinterpret_cast<cuComplex *>(out), rows, cols, by_row);
  }
  else
  {
    dev::vmmul_by_col_long<<<blocks_by_col, threads_by_col>>>
       (kernel,
        reinterpret_cast<cuComplex const*>(in),
        reinterpret_cast<cuComplex*>(out), rows, cols, num_loops);
  }
}

void 
vmmul_col(std::complex<float> const *kernel,
	  float const *in,
	  std::complex<float> *out,
	  length_type rows, length_type cols)
{
  dim3 blocks, threads, blocks_by_col, threads_by_col;
  distribute_matrix(rows, cols, blocks, threads);

  // Create each block with 64 threads and create one block for each row. This
  //  only applies to the "by_col_long" kernel. 
  threads_by_col.x = 64;
  blocks_by_col.x = rows;

  //  Required number of loops for "by_col_long" kernel.
  int num_loops = (cols + threads_by_col.x - 1) / threads_by_col.x;

  bool by_row = false;
  if (rows < 128)
  {
    dev::vmmul<<<blocks, threads>>>
      (reinterpret_cast<cuComplex const*>(kernel),
       in,
       reinterpret_cast<cuComplex*>(out),
       rows, cols, by_row);
  }
  else
  {
    dev::vmmul_by_col_long<<<blocks_by_col, threads_by_col>>>
       (reinterpret_cast<cuComplex const*>(kernel),
        in,
        reinterpret_cast<cuComplex*>(out), rows, cols, num_loops);
  }
}

void 
vmmul_col(std::complex<float> const *kernel,  
	  std::complex<float> const *in,
	  std::complex<float> *out,
	  length_type rows, length_type cols)
{
  dim3 blocks, threads, blocks_by_col, threads_by_col;
  distribute_matrix(rows, cols, blocks, threads);

  // Create each block with 64 threads and create one block for each row. This
  //  only applies to the "by_col_long" kernel. 
  threads_by_col.x = 64;
  blocks_by_col.x = rows;

  //  Required number of loops for "by_col_long" kernel.
  int num_loops = (cols + threads_by_col.x - 1) / threads_by_col.x;

  bool by_row = false;
  if (rows < 128)
  {
    dev::vmmul<<<blocks, threads>>>
      (reinterpret_cast<cuComplex const*>(kernel),
       reinterpret_cast<cuComplex const*>(in),
       reinterpret_cast<cuComplex *>(out), rows, cols, by_row);
  }
  else
  {
    dev::vmmul_by_col_long<<<blocks_by_col, threads_by_col>>>
       (reinterpret_cast<cuComplex const*>(kernel),
        reinterpret_cast<cuComplex const*>(in),
        reinterpret_cast<cuComplex*>(out), rows, cols, num_loops);
  }
}

void 
vmmuls_row(std::complex<float> const *kernel,  
	   std::complex<float> const *in,
	   std::complex<float> *out,
	   float scale,
	   length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  bool by_row = true;
  dev::vmmuls<<<blocks, threads>>>
    (reinterpret_cast<cuComplex const*>(kernel),
     reinterpret_cast<cuComplex const*>(in),
     reinterpret_cast<cuComplex*>(out),
     scale, rows, cols, by_row);
}

void 
vmmuls_col(std::complex<float> const *kernel,  
	   std::complex<float> const *in,
	   std::complex<float> *out,
	   float scale,
	   length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  bool by_row = false;
  dev::vmmuls<<<blocks, threads>>>
    (reinterpret_cast<cuComplex const*>(kernel),
     reinterpret_cast<cuComplex const*>(in),
     reinterpret_cast<cuComplex *>(out),
     scale, rows, cols, by_row);
}


void 
mmmuls(std::complex<float> const *kernel,  
       std::complex<float> const *in,
       std::complex<float> *out,
       float scale,
       length_type rows, length_type cols)
{
  dim3 blocks, threads;
  distribute_matrix(rows, cols, blocks, threads);

  dev::matmuls<<<blocks, threads>>>
    (reinterpret_cast<cuComplex const*>(kernel),
     reinterpret_cast<cuComplex const*>(in),
     reinterpret_cast<cuComplex *>(out),
     scale, rows, cols);
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

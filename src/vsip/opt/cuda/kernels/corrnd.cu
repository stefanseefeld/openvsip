/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA kernels for optimized correlation.
#include <cuComplex.h>
#include <complex>
#include "cmplx.cuh"
#include "util.hpp"

using namespace dev;

// 1-D correlation, shared memory is used to overcome excessive global memory
//  transfers. The entire kernel is read into each block's shared memory so it
//  must fit within a single block's alloted space.
//
//  The result is computed in two stages, first the center elements for which
//  there are no elements of the kernel extending past the edges of the input,
//  and second for the edge cases where the loop iteration count varies.
//
//  The maximum kernel size is 500 elements in the complex case and 512
//  elements in the float casedue to shared memory limitations when doing
//  edge computations.

// Center computation for full support and real data.
__global__ void
k_corrnd_center_full_ss(
  float const* input,
  float const* kernel,
  float*       out,
  int          input_len,
  int          kernel_len,
  int          output_len,
  int          is_biased)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  float sum = 0.0F;

  // Allocate enough memory to hold the input, since each thread computes a
  //  single element of the output any amount > 512 is adequate. 3000 was
  //  chosen arbitrarily with only this constraint in mind.
  __shared__ float input_sh[3000];
  __shared__ float kernel_sh[512];

  kernel_sh[tx] = kernel[tx];
  input_sh[tx] = input[tid];
  input_sh[tx + blockDim.x] = input[tid + blockDim.x];

  __syncthreads();


  // Loops are currently unrolled 8 times.  This number was chosen arbitrarily
  //  for good performance at a range of kernel lengths.  A larger number will
  //  give better performance for larger kernel sizes at the expense of worse
  //  performance for smaller kernels.
  #pragma unroll 8
  for (int i = 0; i < kernel_len; ++i)
  {
    int index = tx + i;
 
    sum += input_sh[index] * kernel_sh[i];
  }

  if (is_biased)
    sum /= float(kernel_len);

  if (tid < input_len - kernel_len + 1)
    *(out + tid + kernel_len - 1) = sum;
}

// Center computation for full support and complex data.
__global__ void
k_corrnd_center_full_cc(
  cuComplex const* input,
  cuComplex const* kernel,
  cuComplex*       out,
  int              input_len,
  int              kernel_len,
  int              output_len,
  int              is_biased)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  cuComplex sum = {0.0F, 0.0F};
  cuComplex temp = {0.0F, 0.0F};

  // Allocate enough memory to hold the input, since each thread computes a
  //  single element of the output any amount > 512 is adequate.1500 ( 3000 from
  //  the float case / 2) was chosen arbitrarily with only this constraint in mind.
  __shared__ cuComplex input_sh[1500];
  __shared__ cuComplex kernel_sh[512];

  kernel_sh[tx] = kernel[tx];
  input_sh[tx] = input[tid];
  input_sh[tx + blockDim.x] = input[tid + blockDim.x];

  __syncthreads();

  // Loops are currently unrolled 8 times.  This number was chosen arbitrarily
  //  for good performance at a range of kernel lengths.  A larger number will
  //  give better performance for larger kernel sizes at the expense of worse
  //  performance for smaller kernels.
  #pragma unroll 8
  for (int i = 0; i < kernel_len; ++i)
  {
    int index = tx + i;
 
    cmulc(temp, kernel_sh[i], input_sh[index]);
    sum.x = __fadd_rn(sum.x, temp.x);
    sum.y = __fadd_rn(sum.y, temp.y);
  }

  if (is_biased)
  {
    sum.x /= float(kernel_len);
    sum.y /= float(kernel_len);
  }

  if (tid < input_len - kernel_len + 1)
    *(out + tid + kernel_len - 1) = sum;
}

// Center computation for same support, odd kernel length, and real data.
__global__ void
k_corrnd_center_same_odd_ss(
  float const* input,
  float const* kernel,
  float*       out,
  int          input_len,
  int          kernel_len,
  int          output_len,
  int          shift,
  int          is_biased)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  float sum = 0.0F;

  // Allocate enough memory to hold the input, since each thread computes a
  //  single element of the output any amount > 512 is adequate.3000 was
  //  chosen arbitrarily with only this constraint in mind.
  __shared__ float input_sh[3000];
  __shared__ float kernel_sh[512];

  kernel_sh[tx] = kernel[tx];
  input_sh[tx] = input[tid];
  input_sh[tx + blockDim.x] = input[tid + blockDim.x];

  __syncthreads();


  // Loops are currently unrolled 8 times.  This number was chosen arbitrarily
  //  for good performance at a range of kernel lengths.  A larger number will
  //  give better performance for larger kernel sizes at the expense of worse
  //  performance for smaller kernels.
  #pragma unroll 8
  for (int i = 0; i < kernel_len; ++i)
  {
    int index = tx + i;
 
    sum += input_sh[index] * kernel_sh[i];
  }

  if (is_biased)
    sum /= float(kernel_len - 1);

  if (tid < input_len - kernel_len + 1)
    *(out + tid + shift) = sum;
}

// Center computation for same support, odd kernel length, and complex data.
__global__ void
k_corrnd_center_same_odd_cc(
  cuComplex const* input,
  cuComplex const* kernel,
  cuComplex*       out,
  int              input_len,
  int              kernel_len,
  int              output_len,
  int              shift,
  int              is_biased)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  cuComplex sum = {0.0F, 0.0F};
  cuComplex temp;

  // Allocate enough memory to hold the input, since each thread computes a
  //  single element of the output any amount > 512 is adequate.1500 ( 3000 from
  //  the float case / 2) was chosen arbitrarily with only this constraint in mind.
  __shared__ cuComplex input_sh[1500];
  __shared__ cuComplex kernel_sh[512];

  kernel_sh[tx] = kernel[tx];
  input_sh[tx] = input[tid];
  input_sh[tx + blockDim.x] = input[tid + blockDim.x];

  __syncthreads();

  // Loops are currently unrolled 8 times.  This number was chosen arbitrarily
  //  for good performance at a range of kernel lengths.  A larger number will
  //  give better performance for larger kernel sizes at the expense of worse
  //  performance for smaller kernels.
  #pragma unroll 8
  for (int i = 0; i < kernel_len; ++i)
  {
    int index = tx + i;
 
    cmulc(temp, kernel_sh[i], input_sh[index]);
    sum.x = __fadd_rn(sum.x, temp.x);
    sum.y = __fadd_rn(sum.y, temp.y);
  }

  if (is_biased)
  {
    sum.x /= float(kernel_len - 1);
    sum.y /= float(kernel_len - 1);
  }

  if (tid < input_len - kernel_len + 1)
    *(out + tid + shift) = sum;
}

// Center computation for same support, even kernel length, and real data.
__global__ void
k_corrnd_center_same_even_ss(
  float const* input,
  float const* kernel,
  float*       out,
  int          input_len,
  int          kernel_len,
  int          output_len,
  int          shift,
  int          is_biased)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  float sum = 0.0F;

  // Allocate enough memory to hold the input, since each thread computes a
  //  single element of the output any amount > 512 is adequate.3000 was
  //  chosen arbitrarily with only this constraint in mind.
  __shared__ float input_sh[3000];
  __shared__ float kernel_sh[512];

  kernel_sh[tx] = kernel[tx];
  input_sh[tx] = input[tid];
  input_sh[tx + blockDim.x] = input[tid + blockDim.x];

  __syncthreads();

  // Loops are currently unrolled 8 times.  This number was chosen arbitrarily
  //  for good performance at a range of kernel lengths.  A larger number will
  //  give better performance for larger kernel sizes at the expense of worse
  //  performance for smaller kernels.
  #pragma unroll 8
  for (int i = 0; i < kernel_len; ++i)
  {
    int index = tx + i;
 
    sum += input_sh[index] * kernel_sh[i];
  }

  if (tid < input_len - kernel_len + 1)
    *(out + tid + shift) = sum;
}

// Center computation for same support, even kernel length, and complex data.
__global__ void
k_corrnd_center_same_even_cc(
  cuComplex const* input,
  cuComplex const* kernel,
  cuComplex*       out,
  int              input_len,
  int              kernel_len,
  int              output_len,
  int              shift,
  int              is_biased)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  cuComplex sum = {0.0F, 0.0F};
  cuComplex temp;

  // Allocate enough memory to hold the input, since each thread computes a
  //  single element of the output any amount > 512 is adequate.1500 ( 3000 from
  //  the float case / 2) was chosen arbitrarily with only this constraint in mind.
  __shared__ cuComplex input_sh[1500];
  __shared__ cuComplex kernel_sh[512];

  kernel_sh[tx] = kernel[tx];
  input_sh[tx] = input[tid];
  input_sh[tx + blockDim.x] = input[tid + blockDim.x];

  __syncthreads();

  // Loops are currently unrolled 8 times.  This number was chosen arbitrarily
  //  for good performance at a range of kernel lengths.  A larger number will
  //  give better performance for larger kernel sizes at the expense of worse
  //  performance for smaller kernels.
  #pragma unroll 8
  for (int i = 0; i < kernel_len; ++i)
  {
    int index = tx + i;
 
    cmulc(temp, kernel_sh[i], input_sh[index]);
    sum.x = __fadd_rn(sum.x, temp.x);
    sum.y = __fadd_rn(sum.y, temp.y);
  }

  if (tid < input_len - kernel_len + 1)
    *(out + tid + shift) = sum;
}

// Center computation for minimum support and real data.
__global__ void
k_corrnd_center_min_ss(
  float const* input,
  float const* kernel,
  float*       out,
  int          input_len,
  int          kernel_len,
  int          output_len,
  int          is_biased)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  float sum = 0.0F;

  // Allocate enough memory to hold the input, since each thread computes a
  //  single element of the output any amount > 512 is adequate.3000 was
  //  chosen arbitrarily with only this constraint in mind.
  __shared__ float input_sh[3000];
  __shared__ float kernel_sh[512];

  kernel_sh[tx] = kernel[tx];
  input_sh[tx] = input[tid];
  input_sh[tx + blockDim.x] = input[tid + blockDim.x];

  __syncthreads();

  // Loops are currently unrolled 8 times.  This number was chosen arbitrarily
  //  for good performance at a range of kernel lengths.  A larger number will
  //  give better performance for larger kernel sizes at the expense of worse
  //  performance for smaller kernels.
  #pragma unroll 8
  for (int i = 0; i < kernel_len; ++i)
  {
    int index = tx + i;

    sum += input_sh[index] * kernel_sh[i];
  }

  if (is_biased)
    sum /= float(kernel_len);

  if (tid < output_len)
    *(out + tid) = sum;
}

// Center computation for minimum support and complex data.
__global__ void
k_corrnd_center_min_cc(
  cuComplex const* input,
  cuComplex const* kernel,
  cuComplex*       out,
  int              input_len,
  int              kernel_len,
  int              output_len,
  int              is_biased)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  cuComplex sum = {0.0F, 0.0F};
  cuComplex temp;

  // Allocate enough memory to hold the input, since each thread computes a
  //  single element of the output any amount > 512 is adequate.1500 ( 3000 from
  //  the float case / 2) was chosen arbitrarily with only this constraint in mind.
  __shared__ cuComplex input_sh[1500];
  __shared__ cuComplex kernel_sh[512];

  kernel_sh[tx] = kernel[tx];
  input_sh[tx] = input[tid];
  input_sh[tx + blockDim.x] = input[tid + blockDim.x];

  __syncthreads();

  // Loops are currently unrolled 8 times.  This number was chosen arbitrarily
  //  for good performance at a range of kernel lengths.  A larger number will
  //  give better performance for larger kernel sizes at the expense of worse
  //  performance for smaller kernels.
  #pragma unroll 8
  for (int i = 0; i < kernel_len; ++i)
  {
    int index = tx + i;

    cmulc(temp, kernel_sh[i], input_sh[index]);
    sum.x = __fadd_rn(sum.x, temp.x);
    sum.y = __fadd_rn(sum.y, temp.y);
  }

  if (is_biased)
  {
    sum.x /= float(kernel_len);
    sum.y /= float(kernel_len);
  }

  if (tid < output_len)
    *(out + tid) = sum;
}

// Edge computation for full support and real data.
__global__ void
k_corrnd_edge_full_ss(
  float const* input,
  float const* kernel,
  float*       out,
  int          input_len,
  int          kernel_len,
  int          output_len,
  int          is_biased)
{
  int tx = threadIdx.x;

  float sum_begin = 0.0F;
  float sum_end = 0.0F;

  // Allocate enough memory to hold the input and kernels.  Each block has 512
  //  threads and computes a single element at the beginning and end of the
  //  input, thus each block requires 512*2 floats in shared memory for the
  //  kernel and input.
  __shared__ float input_sh[1024];
  __shared__ float kernel_sh[512];


  input_sh[tx] = input[tx];
  input_sh[tx + kernel_len] = input[input_len - kernel_len + tx];
  kernel_sh[tx] = kernel[tx];

  __syncthreads();

  #pragma unroll 8
  for (int i = 0; i <= tx; ++i)
  {
    sum_begin += input_sh[i] * kernel_sh[i + kernel_len - 1 - tx];
    sum_end += input_sh[__mul24(kernel_len, 2) - tx + i - 1] *
               kernel_sh[i];
  }

  if (is_biased)
  {
    sum_begin /= float(tx + 1);
    sum_end /= float(tx + 1);
  }

  if (tx < kernel_len - 1)
  {
    *(out + tx) = sum_begin;
    *(out + output_len - tx - 1) = sum_end;
  }

}

// Edge computation for full support and complex data.
__global__ void
k_corrnd_edge_full_cc(
  cuComplex const* input,
  cuComplex const* kernel,
  cuComplex*       out,
  int              input_len,
  int              kernel_len,
  int              output_len,
  int              is_biased)
{
  int tx = threadIdx.x;

  cuComplex sum_begin = {0.0F, 0.0F};
  cuComplex temp_begin;
  cuComplex sum_end = {0.0F, 0.0F};
  cuComplex temp_end;

  // Allocate enough memory to hold the input and kernels.  Each block has 512
  //  threads and computes a single element at the beginning and end of the
  //  input, thus each block requires 512*2 floats in shared memory for the
  //  kernel and input.
  // For the complex case each element is 8 bytes thus for 512 elements each 
  //  block requires 512*2*8 bytes = 8192 bytes for each input and kernel.
  //  This requires a total of 16384 bytes of shared memory + the input 
  //  parameters.  The kernel size must be limited appropriately, in this case to
  //  510 elements, thus 510*2 = 1020 elements in shared memory for the input
  //  and the kernel.
  __shared__ cuComplex input_sh[1020];
  __shared__ cuComplex kernel_sh[510];


  input_sh[tx] = input[tx];
  input_sh[tx + kernel_len] = input[input_len - kernel_len + tx];
  kernel_sh[tx] = kernel[tx];

  __syncthreads();

  #pragma unroll 8
  for (int i = 0; i <= tx; ++i)
  {
    cmulc(temp_begin, kernel_sh[i + kernel_len - 1 - tx], input_sh[i]);
    cmulc(temp_end, kernel_sh[i], input_sh[__mul24(kernel_len, 2) - tx + i - 1]);

      sum_begin.x = __fadd_rn(sum_begin.x, temp_begin.x);
      sum_begin.y = __fadd_rn(sum_begin.y, temp_begin.y);

      sum_end.x = __fadd_rn(sum_end.x, temp_end.x);
      sum_end.y = __fadd_rn(sum_end.y, temp_end.y);
  }

  if (is_biased)
  {
    sum_begin.x /= float(tx + 1);
    sum_begin.y /= float(tx + 1);

    sum_end.x /= float(tx + 1);
    sum_end.y /= float(tx + 1);
  }

  if (tx < kernel_len - 1)
  {
    *(out + tx) = sum_begin;
    *(out + output_len - tx - 1) = sum_end;
  }
}

// Edge computation for same support, odd kernel length, and real data.
__global__ void
k_corrnd_edge_same_odd_ss(
  float const* input,
  float const* kernel,
  float*       out,
  int          input_len,
  int          kernel_len,
  int          output_len,
  int          shift,
  int          is_biased)
{
  int tx = threadIdx.x;

  float sum_begin = 0.0F;
  float sum_end = 0.0F;

  // Allocate enough memory to hold the input and kernels.  Each block has 512
  //  threads and computes a single element at the beginning and end of the
  //  input, thus each block requires 512*2 floats in shared memory for the
  //  kernel and input.
  __shared__ float input_sh[1024];
  __shared__ float kernel_sh[512];


  input_sh[tx] = input[tx];
  input_sh[tx + kernel_len] = input[input_len - kernel_len + tx];
  kernel_sh[tx] = kernel[tx];

  __syncthreads();

  #pragma unroll 8
  for (int i = 0; i <= tx + shift; ++i)
  {
    sum_begin += input_sh[i] * kernel_sh[i + shift - tx];
    sum_end += input_sh[__mul24(kernel_len, 2) - tx + i - 1 - shift] *
               kernel_sh[i];
  }

  if (tx < kernel_len - 1 - shift)
  {
    *(out + tx) = sum_begin;
    *(out + output_len - tx - 1) = sum_end;
  }

}

// Edge computation for same support, odd kernel length, and complex data.
__global__ void
k_corrnd_edge_same_odd_cc(
  cuComplex const* input,
  cuComplex const* kernel,
  cuComplex*       out,
  int              input_len,
  int              kernel_len,
  int              output_len,
  int              shift,
  int              is_biased)
{
  int tx = threadIdx.x;

  cuComplex sum_begin = {0.0F, 0.0F};
  cuComplex temp_begin;
  cuComplex sum_end = {0.0F, 0.0F};
  cuComplex temp_end;

  // Allocate enough memory to hold the input and kernels.  Each block has 512
  //  threads and computes a single element at the beginning and end of the
  //  input, thus each block requires 512*2 floats in shared memory for the
  //  kernel and input.
  // For the complex case each element is 8 bytes thus for 512 elements each 
  //  block requires 512*2*8 bytes = 8192 bytes for each input and kernel.
  //  This requires a total of 16384 bytes of shared memory + the input 
  //  parameters.  The kernel size must be limited appropriately, in this case to
  //  510 elements, thus 510*2 = 1020 elements in shared memory for the input
  //  and the kernel.
  __shared__ cuComplex input_sh[1020];
  __shared__ cuComplex kernel_sh[510];


  input_sh[tx] = input[tx];
  input_sh[tx + kernel_len] = input[input_len - kernel_len + tx];
  kernel_sh[tx] = kernel[tx];

  __syncthreads();

  #pragma unroll 8
  for (int i = 0; i <= tx + shift; ++i)
  {
    cmulc(temp_begin, kernel_sh[i + shift - tx], input_sh[i]);
    cmulc(temp_end, kernel_sh[i], input_sh[__mul24(kernel_len, 2) - tx + i - 1 - shift]);

      sum_begin.x = __fadd_rn(sum_begin.x, temp_begin.x);
      sum_begin.y = __fadd_rn(sum_begin.y, temp_begin.y);

      sum_end.x = __fadd_rn(sum_end.x, temp_end.x);
      sum_end.y = __fadd_rn(sum_end.y, temp_end.y);
  }

  if (tx < kernel_len - 1 - shift)
  {
    *(out + tx) = sum_begin;
    *(out + output_len - tx - 1) = sum_end;
  }
}

// Edge computation for same support, even kernel length, and real data.
__global__ void
k_corrnd_edge_same_even_ss(
  float const* input,
  float const* kernel,
  float*       out,
  int          input_len,
  int          kernel_len,
  int          output_len,
  int          shift,
  int          is_biased)
{
  int tx = threadIdx.x;

  float sum_begin = 0.0F;
  float sum_end = 0.0F;

  // Allocate enough memory to hold the input and kernels.  Each block has 512
  //  threads and computes a single element at the beginning and end of the
  //  input, thus each block requires 512*2 floats in shared memory for the
  //  kernel and input.
  __shared__ float input_sh[1024];
  __shared__ float kernel_sh[512];

  input_sh[tx] = input[tx];
  input_sh[tx + kernel_len] = input[input_len - kernel_len + tx];
  kernel_sh[tx] = kernel[tx];

  __syncthreads();

  #pragma unroll 8
  for (int i = 0; i <= tx + shift - 1; ++i)
  {
    sum_begin += input_sh[i] * kernel_sh[i + shift - tx];
    sum_end += input_sh[__mul24(kernel_len, 2) - tx + i - shift] *
               kernel_sh[i];
  }

  if (tx == 0)
    *(out) = sum_begin;
  else if (tx < kernel_len - shift && tx > 0)
  {
    *(out + tx) = sum_begin;
    *(out + output_len - tx) = sum_end;
  }
}

// Edge computation for same support, even kernel length, and complex data.
__global__ void
k_corrnd_edge_same_even_cc(
  cuComplex const* input,
  cuComplex const* kernel,
  cuComplex*       out,
  int              input_len,
  int              kernel_len,
  int              output_len,
  int              shift,
  int              is_biased)
{
  int tx = threadIdx.x;

  cuComplex sum_begin = {0.0F, 0.0F};
  cuComplex temp_begin;
  cuComplex sum_end = {0.0F, 0.0F};
  cuComplex temp_end;

  // Allocate enough memory to hold the input and kernels.  Each block has 512
  //  threads and computes a single element at the beginning and end of the
  //  input, thus each block requires 512*2 floats in shared memory for the
  //  kernel and input.
  // For the complex case each element is 8 bytes thus for 512 elements each 
  //  block requires 512*2*8 bytes = 8192 bytes for each input and kernel.
  //  This requires a total of 16384 bytes of shared memory + the input 
  //  parameters.  The kernel size must be limited appropriately, in this case to
  //  510 elements, thus 510*2 = 1020 elements in shared memory for the input
  //  and the kernel.
  __shared__ cuComplex input_sh[1020];
  __shared__ cuComplex kernel_sh[510];


  input_sh[tx] = input[tx];
  input_sh[tx + kernel_len] = input[input_len - kernel_len + tx];
  kernel_sh[tx] = kernel[tx];

  __syncthreads();

  #pragma unroll 8
  for (int i = 0; i <= tx + shift - 1; ++i)
  {
    cmulc(temp_begin, kernel_sh[i + shift - tx], input_sh[i]);
    cmulc(temp_end, kernel_sh[i], input_sh[__mul24(kernel_len, 2) - tx + i - shift]);

      sum_begin.x = __fadd_rn(sum_begin.x, temp_begin.x);
      sum_begin.y = __fadd_rn(sum_begin.y, temp_begin.y);

      sum_end.x = __fadd_rn(sum_end.x, temp_end.x);
      sum_end.y = __fadd_rn(sum_end.y, temp_end.y);
  }

  if (tx == 0)
    *(out) = sum_begin;
  else if (tx < kernel_len - shift && tx > 0)
  {
    *(out + tx) = sum_begin;
    *(out + output_len - tx) = sum_end;
  }
}

namespace vsip
{
namespace impl
{
namespace cuda
{

void
corr_no_decimation_full(
  float const*     in,
  float const*     kr,
  float*           out,
  size_t           in_len,
  size_t           kr_len,
  size_t           ou_len,
  int              bias)
{
  dim3 grid, threads, grid_edge, threads_edge;
  distribute_vector(in_len, grid, threads);
  grid_edge.x = 1;
  threads_edge.x = kr_len;

  k_corrnd_center_full_ss<<<grid, threads>>>(in, kr, out, int(in_len),
                                             int(kr_len), int(ou_len), bias);

  k_corrnd_edge_full_ss<<<grid_edge, threads_edge>>>(in, kr, out, int(in_len),
                                                     int(kr_len), int(ou_len), bias);

}

void
corr_no_decimation_full(
  std::complex<float> const*     in,
  std::complex<float> const*     kr,
  std::complex<float>*           out,
  size_t                         in_len,
  size_t                         kr_len,
  size_t                         ou_len,
  int                            bias)
{
  dim3 grid, threads, grid_edge, threads_edge;
  distribute_vector(in_len, grid, threads);
  grid_edge.x = 1;
  threads_edge.x = kr_len;

  k_corrnd_center_full_cc<<<grid, threads>>>(
    reinterpret_cast<cuComplex const*>(in),
    reinterpret_cast<cuComplex const*>(kr),
    reinterpret_cast<cuComplex*>(out), int(in_len),
    int(kr_len), int(ou_len), bias);

  k_corrnd_edge_full_cc<<<grid_edge, threads_edge>>>(
    reinterpret_cast<cuComplex const*>(in),
    reinterpret_cast<cuComplex const*>(kr),
    reinterpret_cast<cuComplex*>(out), int(in_len),
    int(kr_len), int(ou_len), bias);


}

void
corr_no_decimation_min(
  float const*     in,
  float const*     kr,
  float*           out,
  size_t           in_len,
  size_t           kr_len,
  size_t           ou_len,
  int              bias)
{
  dim3 grid, threads;
  distribute_vector(in_len, grid, threads);

  k_corrnd_center_min_ss<<<grid, threads>>>(in, kr, out, int(in_len),
                                            int(kr_len), int(ou_len), bias);

}

void
corr_no_decimation_min(
  std::complex<float> const*     in,
  std::complex<float> const*     kr,
  std::complex<float>*           out,
  size_t                         in_len,
  size_t                         kr_len,
  size_t                         ou_len,
  int                            bias)
{
  dim3 grid, threads;
  distribute_vector(in_len, grid, threads);

  k_corrnd_center_min_cc<<<grid, threads>>>(
    reinterpret_cast<cuComplex const*>(in),
    reinterpret_cast<cuComplex const*>(kr),
    reinterpret_cast<cuComplex*>(out), int(in_len),
    int(kr_len), int(ou_len), bias);

}

void
corr_no_decimation_same_odd(
  float const*     in,
  float const*     kr,
  float*           out,
  size_t           in_len,
  size_t           kr_len,
  size_t           ou_len,
  size_t           shift,
  int              bias)
{
  dim3 grid, threads, grid_edge, threads_edge;
  distribute_vector(in_len, grid, threads);
  grid_edge.x = 1;
  threads_edge.x = kr_len;

  k_corrnd_center_same_odd_ss<<<grid, threads>>>(in, kr, out, int(in_len),
                                                 int(kr_len), int(ou_len),
                                                 int(shift), bias);

  k_corrnd_edge_same_odd_ss<<<grid_edge, threads_edge>>>(in, kr, out, int(in_len),
                                                         int(kr_len), int(ou_len),
                                                         int(shift), bias);

}

void
corr_no_decimation_same_odd(
  std::complex<float> const*     in,
  std::complex<float> const*     kr,
  std::complex<float>*           out,
  size_t                         in_len,
  size_t                         kr_len,
  size_t                         ou_len,
  size_t                         shift,
  int                            bias)
{
  dim3 grid, threads, grid_edge, threads_edge;
  distribute_vector(in_len, grid, threads);
  grid_edge.x = 1;
  threads_edge.x = kr_len;

  k_corrnd_center_same_odd_cc<<<grid, threads>>>(
    reinterpret_cast<cuComplex const*>(in),
    reinterpret_cast<cuComplex const*>(kr),
    reinterpret_cast<cuComplex*>(out), int(in_len),
    int(kr_len), int(ou_len), int(shift), bias);

  k_corrnd_edge_same_odd_cc<<<grid_edge, threads_edge>>>(
    reinterpret_cast<cuComplex const*>(in),
    reinterpret_cast<cuComplex const*>(kr),
    reinterpret_cast<cuComplex*>(out), int(in_len),
    int(kr_len), int(ou_len), int(shift), bias);


}

void
corr_no_decimation_same_even(
  float const*     in,
  float const*     kr,
  float*           out,
  size_t           in_len,
  size_t           kr_len,
  size_t           ou_len,
  size_t           shift,
  int              bias)
{
  dim3 grid, threads, grid_edge, threads_edge;
  distribute_vector(in_len, grid, threads);
  grid_edge.x = 1;
  threads_edge.x = kr_len;

  k_corrnd_center_same_even_ss<<<grid, threads>>>(in, kr, out, int(in_len),
                                                  int(kr_len), int(ou_len),
                                                  int(shift), bias);

  k_corrnd_edge_same_even_ss<<<grid_edge, threads_edge>>>(in, kr, out, int(in_len),
                                                          int(kr_len), int(ou_len),
                                                          int(shift), bias);

}

void
corr_no_decimation_same_even(
  std::complex<float> const*     in,
  std::complex<float> const*     kr,
  std::complex<float>*           out,
  size_t                         in_len,
  size_t                         kr_len,
  size_t                         ou_len,
  size_t                         shift,
  int                            bias)
{
  dim3 grid, threads, grid_edge, threads_edge;
  distribute_vector(in_len, grid, threads);
  grid_edge.x = 1;
  threads_edge.x = kr_len;

  k_corrnd_center_same_even_cc<<<grid, threads>>>(
    reinterpret_cast<cuComplex const*>(in),
    reinterpret_cast<cuComplex const*>(kr),
    reinterpret_cast<cuComplex*>(out), int(in_len),
    int(kr_len), int(ou_len), int(shift), bias);

  k_corrnd_edge_same_even_cc<<<grid_edge, threads_edge>>>(
    reinterpret_cast<cuComplex const*>(in),
    reinterpret_cast<cuComplex const*>(kr),
    reinterpret_cast<cuComplex*>(out), int(in_len),
    int(kr_len), int(ou_len), int(shift), bias);


}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

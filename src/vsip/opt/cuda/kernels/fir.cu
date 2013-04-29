/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA kernels for FIR filter with unit decimation.
#include <cuComplex.h>
#include <complex>
#include "cmplx.cuh"
#include "util.hpp"

using namespace dev;


// Main FIR computation without state knowledge for real data
__global__ void
fir_center(
  float const* input,
  float const* kernel,
  float*       out,
  int          kernel_len,
  int          output_len)
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
    int index = tx - i + kernel_len - 1;

    sum += input_sh[index] * kernel_sh[i];
  }

  if (tid < output_len - kernel_len + 1)
    *(out + tid + kernel_len - 1) = sum;
}

// Main FIR computation without state knowledge for complex data
__global__ void
fir_center(
  cuComplex const* input,
  cuComplex const* kernel,
  cuComplex*       out,
  int              kernel_len,
  int              output_len)
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
    int index = tx - i + kernel_len - 1;

    cmul(temp, input_sh[index], kernel_sh[i]);
    sum.x = __fadd_rn(sum.x, temp.x);
    sum.y = __fadd_rn(sum.y, temp.y);
  }

  if (tid < output_len - kernel_len + 1)
    *(out + tid + kernel_len - 1) = sum;
}

// Initial values using the state vector for real data
__global__ void
fir_state(
  float const* input,
  float*       state,
  float const* kernel,
  float*       out,
  int          kernel_len,
  int          output_len)
{
  int tx = threadIdx.x;

  if (tx >= kernel_len || tx >= output_len)
    return;

  float sum = 0.0F;

  // Allocate enough memory to hold the input and kernels.  Each block has 512
  //  threads and computes a single element at the beginning and end of the
  //  input, thus each block requires 512*2 floats in shared memory for the
  //  kernel and input.
  __shared__ float input_sh[1024];
  __shared__ float kernel_sh[512];


  input_sh[tx] = state[tx];
  input_sh[tx + kernel_len - 1] = input[tx];
  kernel_sh[tx] = kernel[tx];

  __syncthreads();

  #pragma unroll 8
  for (int i = 0; i < kernel_len; ++i)
    sum += input_sh[tx + kernel_len - 1 - i] * kernel_sh[i];

  *(out + tx) = sum;

  // Capture the remaining states into the state storage for later use.
  if (tx < kernel_len - 1)
    *(state + tx) = input[output_len - kernel_len + 1 + tx];
}

// Initial values using the state vector for complex data
__global__ void
fir_state(
  cuComplex const* input,
  cuComplex*       state,
  cuComplex const* kernel,
  cuComplex*       out,
  int              kernel_len,
  int              output_len)
{
  int tx = threadIdx.x;

  if (tx >= kernel_len || tx >= output_len)
    return;

  cuComplex sum = {0.0F, 0.0F};
  cuComplex temp;

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


  input_sh[tx] = state[tx];
  input_sh[tx + kernel_len - 1] = input[tx];
  kernel_sh[tx] = kernel[tx];

  __syncthreads();

  #pragma unroll 8
  for (int i = 0; i < kernel_len; ++i)
  {
    cmul(temp, input_sh[tx + kernel_len - 1 - i], kernel_sh[i]);

    sum.x = __fadd_rn(sum.x, temp.x);
    sum.y = __fadd_rn(sum.y, temp.y);
  }

  *(out + tx) = sum;

  // Capture the remaining states into the state storage for later use.
  if (tx < kernel_len - 1)
    *(state + tx) = input[output_len - kernel_len + 1 + tx];

}
namespace vsip
{
namespace impl
{
namespace cuda
{

void
fir_no_decimation(
  float const*     in,
  float*           out,
  size_t           ou_len,
  float const*     kr,
  size_t           kr_len,
  float*           st)
{
  dim3 grid, threads, grid_state, threads_state;
  distribute_vector(ou_len, grid, threads);
  distribute_vector(kr_len - 1, grid_state, threads_state);

  fir_state<<<grid_state, threads_state>>>(in, st, kr, out, int(kr_len), int(ou_len));

  fir_center<<<grid, threads>>>(in, kr, out, int(kr_len), int(ou_len));

}

void
fir_no_decimation(
  std::complex<float> const*     in,
  std::complex<float>*           out,
  size_t                         ou_len,
  std::complex<float> const*     kr,
  size_t                         kr_len,
  std::complex<float>*           st)
{
  dim3 grid, threads, grid_state, threads_state;
  distribute_vector(ou_len, grid, threads);
  distribute_vector(kr_len - 1, grid_state, threads_state);

  fir_state<<<grid_state, threads_state>>>(
    reinterpret_cast<cuComplex const*>(in),
    reinterpret_cast<cuComplex*>(st),
    reinterpret_cast<cuComplex const*>(kr),
    reinterpret_cast<cuComplex*>(out),
    int(kr_len), int(ou_len));

  fir_center<<<grid, threads>>>(
    reinterpret_cast<cuComplex const*>(in),
    reinterpret_cast<cuComplex const*>(kr),
    reinterpret_cast<cuComplex*>(out),
    int(kr_len), int(ou_len));

}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

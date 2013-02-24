/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA kernels for reduction operators and index reductions.  All
///   calculations are performed on a single block to eliminate inter-block
///   thread communication.  The number of elements accumulated per thread
///   is allowed to vary in order to keep the number of required threads <= the
///   maximum number of threads per block.  The calculation then proceeds in
///   log_b(length) stages where b is the number of elements added per thread.
#include <cuComplex.h>
#include <complex>
#include "util.hpp"
#include "cmplx.cuh"

using namespace dev;

// Types of reduction desired: sum, sum of squares, and mean.
typedef enum reductions
{
  reduce_all_true,
  reduce_all_true_bool,
  reduce_any_true,
  reduce_any_true_bool,
  reduce_mean,
  reduce_mean_magsq,
  reduce_sum,
  reduce_sum_bool,
  reduce_sum_sq,
  reduce_max_magsq,
  reduce_max_mag,
  reduce_min_magsq,
  reduce_min_mag,
  reduce_max,
  reduce_min
} reduce_type;

// Compute the integer result base^in
__device__ int
d_powi_ii(int in, int base)
{
  int out = 1;
  for (int i = 0; i < in; ++i)
    out = __mul24(out, base);

  return(out);
}

// Parallel copy of input data in global memory to temporary storage in global
//  memory.  The data is also scaled by the appropriate factor in order to 
//  calculate the proper reduction based on 'cmd'.  Each thread copies
//  'chunk_size' elements from 'in' to 'out'.
__device__ void
d_reduce_copy_ss(float const* in,
                 float*       out,
                 size_t       length,
                 size_t       chunk_size,
                 int          id,
                 reduce_type  cmd)
{
  for (int i = 0; i < chunk_size; ++i)
  {
    if (i + __mul24(id, chunk_size) < length)
    {
      if (cmd == reduce_sum || cmd == reduce_max || cmd == reduce_min)
      {
        *(out + i + __mul24(id, chunk_size)) = *(in + i +
                                                 __mul24(id, chunk_size));
      }
      else if (cmd == reduce_sum_sq || cmd == reduce_max_magsq || cmd == reduce_min_magsq)
      {
        *(out + i + __mul24(id, chunk_size)) = (*(in + i +
                                     __mul24(id, chunk_size))) *
                                    (*(in + i + __mul24(id, chunk_size)));
      }
      else if (cmd == reduce_max_mag || cmd == reduce_min_mag)
      {
        *(out + i + __mul24(id, chunk_size)) = fabsf(*(in + i +
                                                    __mul24(id, chunk_size)));
      }
      else if (cmd == reduce_mean)
      {
        *(out + i + __mul24(id, chunk_size)) = (*(in + i +
                                     __mul24(id, chunk_size))) / (float)length;
      }
      else if (cmd == reduce_mean_magsq)
      {
        *(out + i + __mul24(id, chunk_size)) = (*(in + i +
                                     __mul24(id, chunk_size))) * (*(in + i +
                                     __mul24(id, chunk_size))) / (float)length;
      }
    }
  }
}

__device__ void
d_reduce_copy_cc(cuComplex const* in,
                 cuComplex*       out,
                 size_t           length,
                 size_t           chunk_size,
                 int              id,
                 reduce_type      cmd)
{
  for (int i = 0; i < chunk_size; ++i)
  {
    if (i + __mul24(id, chunk_size) < length)
    {
      if (cmd == reduce_sum)
        *(out + i + __mul24(id, chunk_size)) = *(in + i +
                                                 __mul24(id, chunk_size));
      else if (cmd == reduce_sum_sq)
        cmul(*(out + i + __mul24(id, chunk_size)),
             *(in + i + __mul24(id, chunk_size)),
             *(in + i + __mul24(id, chunk_size)));
      else if (cmd == reduce_mean)
        cdivr(*(out + i + __mul24(id, chunk_size)),
              *(in + i + __mul24(id, chunk_size)),
              (float)length);
    }
  }
}

__device__ void
d_reduce_magsq_copy_cs(cuComplex const* in,
                       float*           out,
                       size_t           length,
                       size_t           chunk_size,
                       int              id,
                       reduce_type      cmd)
{
  for (int i = 0; i < chunk_size; ++i)
  {
    if (i + __mul24(id, chunk_size) < length)
    {
      cmagsq(*(out + i + __mul24(id, chunk_size)),
             *(in + i + __mul24(id, chunk_size)));

      if (cmd == reduce_mean_magsq)
        *(out + i + __mul24(id, chunk_size)) /= (float)length;
      else if (cmd == reduce_max_mag || cmd == reduce_min_mag)
        *(out + i + __mul24(id, chunk_size)) = sqrtf(*(out + i +
                                                    __mul24(id, chunk_size)));
    }
  }
}

// Accumulation of results using 'chunk_size' elements per thread. Each thread
//  writes the result of each stage ('nstages' total) to it's own initial
//  memory location.  'id' is the ID of the thread calling this function.
//  'stride' is the stride to each successive element to include in the 
//  reduction.  Accumulation may be done by summation, maximum, or minimum
//  and may further depend on the input/output types.
__device__ void
d_reduce_accum_ss(float*     data,
                  size_t     length,
                  ptrdiff_t  stride,
                  size_t     nstages,
                  size_t     chunk_size,
                  int        id)
{
  for (int i = 0; i < nstages; ++i)
  {
    int index = __mul24(d_powi_ii(i + 1, chunk_size), id);
    int skip  = d_powi_ii(i, chunk_size);
    
    // Verify that the thread is writing to a valid location.
    if (__mul24(index, stride) < length)
    {
      for (int j = 1; j < chunk_size; ++j)
        data[__mul24(index, stride)] += (__mul24((index + __mul24(skip, j)), stride) < length ?
                                 data[__mul24((index + __mul24(skip, j)), stride)] : 0);
    }
    __syncthreads();
  }
}

__device__ void
d_reduce_accum_cc(cuComplex * data,
                  size_t      length,
                  ptrdiff_t   stride,
                  size_t      nstages,
                  size_t      chunk_size,
                  int         id)
{
  for (int i = 0; i < nstages; ++i)
  {
    int index = __mul24(d_powi_ii(i + 1, chunk_size), id);
    int skip  = d_powi_ii(i, chunk_size);
    
    // Verify that the thread is writing to a valid location.
    if (__mul24(index, stride) < length)
    {
      for (int j = 1; j < chunk_size; ++j)
      {
        data[__mul24(index, stride)].x += (__mul24((index + __mul24(skip, j)), stride) < length ?
                                   data[__mul24((index + __mul24(skip, j)), stride)].x : 0);
        data[__mul24(index, stride)].y += (__mul24((index + __mul24(skip, j)), stride) < length ?
                                   data[__mul24((index + __mul24(skip, j)), stride)].y : 0);
      }
    }
    __syncthreads();
  }
}

// Maximum and Minimum accumulations also accumulate the index values in 'idx'
//  corresponding to the maximum or minimum value in 'data' for each calculation
__device__ void
d_reduce_accum_max_ss(float*     data,
                      size_t*    idx,
                      size_t     length,
                      ptrdiff_t  stride,
                      size_t     nstages,
                      size_t     chunk_size,
                      int        id)
{
  if (id == 0)
  {
    for (int i = 0; i < length; ++i)
      idx[i] = i;
  }
  __syncthreads();

  for (int i = 0; i < nstages; ++i)
  {
    int index = __mul24(d_powi_ii(i + 1, chunk_size), id);
    int skip  = d_powi_ii(i, chunk_size);
    
    // Verify that the thread is writing to a valid location.
    if (__mul24(index, stride) < length)
    {
      for (int j = 1; j <= chunk_size; ++j)
      {
        if (__mul24((index + __mul24(skip, j)), stride) < length)
        {
          if (data[__mul24((index + __mul24(skip, j)), stride)] > data[__mul24(index, stride)])
          {
            data[__mul24(index, stride)] = data[__mul24((index + __mul24(skip, j)), stride)];
            idx[__mul24(index, stride)] = idx[__mul24((index + __mul24(skip, j)), stride)];
          }
        }
      }
    }
    __syncthreads();
  }
}

__device__ void
d_reduce_accum_min_ss(float*     data,
                      size_t*    idx,
                      size_t     length,
                      ptrdiff_t  stride,
                      size_t     nstages,
                      size_t     chunk_size,
                      int        id)
{
  if (id == 0)
  {
    for (int i = 0; i < length; ++i)
      idx[i] = i;
  }
  __syncthreads();

  for (int i = 0; i < nstages; ++i)
  {
    int index = __mul24(d_powi_ii(i + 1, chunk_size), id);
    int skip  = d_powi_ii(i, chunk_size);
    
    // Verify that the thread is writing to a valid location.
    if (__mul24(index, stride) < length)
    {
      for (int j = 1; j <= chunk_size; ++j)
      {
        if (__mul24((index + __mul24(skip, j)), stride) < length)
        {
          if (data[__mul24((index + __mul24(skip, j)), stride)] < data[__mul24(index, stride)])
          {
            data[__mul24(index, stride)] = data[__mul24((index + __mul24(skip, j)), stride)];
            idx[__mul24(index, stride)] = idx[__mul24((index + __mul24(skip, j)), stride)];
          }
        }
      }                       
    }
    __syncthreads();
  }
}

// Kernel to perform a reduction based on the type specified by 'reduce_type'.
//  'space' is a temporary memory location to copy scaled
//  data ('nrows' x 'ncols') to while performing the reduction.
//  'nsteps' is the number of stages to use while reducing using a
//  radix of 'radix'.  For index reductions index values are reduced in 
//  'space_idx' and stored in 'row_idx' and 'col_idx' as appropriate.
__global__ void
k_reduce_cc(cuComplex const* input,
            cuComplex*       space,
            cuComplex*       sum,
	    size_t           nrows,
            size_t           ncols,
            size_t           nsteps,
	    size_t           radix,
            reduce_type      type)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  if (tid == 0)
  {
    sum[0].x = 0.0;
    sum[0].y = 0.0;
  }

  for (int i = 0; i < nrows; ++i)
  {
    d_reduce_copy_cc(input + __mul24(ncols, i), space, ncols, radix, tid,
                                     type);

    __syncthreads();

    d_reduce_accum_cc(space, ncols, 1, nsteps, radix, tid);

    if (tid == 0)
    {
      if (type == reduce_mean)
      {
        sum[0].x += space[0].x / (float)nrows;
        sum[0].y += space[0].y / (float)nrows;
      }
      else
      {
        sum[0].x += space[0].x;
        sum[0].y += space[0].y;
      }
    }
  }
}

__global__ void
k_reduce_ss(float const* input,
            float*       space,
            float*       sum,
            size_t*      space_idx,
            size_t*      row_idx,
            size_t*      col_idx,
	    size_t       nrows,
            size_t       ncols,
            size_t       nsteps,
            size_t       radix,
            reduce_type  type)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  if (tid == 0)
    sum[0] = 0.0;


  d_reduce_copy_ss(input + __mul24(ncols, 0), space, ncols, radix, tid,
                   type);

  __syncthreads();

  if (type == reduce_max || type == reduce_max_magsq || type == reduce_max_mag)
    d_reduce_accum_max_ss(space, space_idx, ncols, 1, nsteps, radix, tid);
  else if (type == reduce_sum || type == reduce_sum_sq || type == reduce_mean || type == reduce_mean_magsq)
    d_reduce_accum_ss(space, ncols, 1, nsteps, radix, tid);
  else if (type == reduce_min || type == reduce_min_magsq || type == reduce_min_mag)
    d_reduce_accum_min_ss(space, space_idx, ncols, 1, nsteps, radix, tid);

  if (tid == 0)
  {
    if (type == reduce_mean || type == reduce_mean_magsq)
      sum[0] += space[0] / float(nrows);
    else if (type == reduce_sum || type == reduce_sum_sq)
      sum[0] += space[0];
    else if (type == reduce_max_magsq || type == reduce_max || type == reduce_max_mag)
    {
      sum[0] = space[0];
      *row_idx = 0;
      *col_idx = space_idx[0];
    }
    else if (type == reduce_min_magsq || type == reduce_min || type == reduce_min_mag)
    {
      sum[0] = space[0];
      *row_idx = 0;
      *col_idx = space_idx[0];
    }
  }

  __syncthreads();


  for (int i = 1; i < nrows; ++i)
  {
    d_reduce_copy_ss(input + __mul24(ncols, i), space, ncols, radix, tid,
                     type);

    __syncthreads();

    if (type == reduce_max || type == reduce_max_magsq || type == reduce_max_mag)
      d_reduce_accum_max_ss(space, space_idx, ncols, 1, nsteps, radix, tid);
    else if (type == reduce_sum || type == reduce_sum_sq || type == reduce_mean || type == reduce_mean_magsq)
      d_reduce_accum_ss(space, ncols, 1, nsteps, radix, tid);
    else if (type == reduce_min || type == reduce_min_magsq || type == reduce_min_mag)
      d_reduce_accum_min_ss(space, space_idx, ncols, 1, nsteps, radix, tid);

    if (tid == 0)
    {
      if (type == reduce_mean || type == reduce_mean_magsq)
        sum[0] += space[0] / float(nrows);
      else if (type == reduce_sum || type == reduce_sum_sq)
        sum[0] += space[0];
      else if (type == reduce_max_magsq || type == reduce_max || type == reduce_max_mag)
      {
        if (space[0] > sum[0])
        {
          sum[0] = space[0];
          *row_idx = i;
          *col_idx = space_idx[0];
        }
      }
      else if (type == reduce_min_magsq || type == reduce_min || type == reduce_min_mag)
      {
        if (space[0] < sum[0])
        {
          sum[0] = space[0];
          *row_idx = i;
          *col_idx = space_idx[0];
        }
      }
    }
    __syncthreads();
  }
}

__global__ void
k_reduce_magsq_cs(cuComplex const* input,
                  float*       space,
                  float*       sum,
                  size_t*      space_idx,
                  size_t*      row_idx,
                  size_t*      col_idx,
	          size_t       nrows,
                  size_t       ncols,
                  size_t       nsteps,
                  size_t       radix,
                  reduce_type  type)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;

  if (tid == 0)
    sum[0] = 0.0;

  d_reduce_magsq_copy_cs(input + __mul24(ncols, 0), space, ncols, radix, tid, type);

  __syncthreads();

  if (type == reduce_max_magsq || type == reduce_max_mag)
    d_reduce_accum_max_ss(space, space_idx, ncols, 1, nsteps, radix, tid);
  else if (type == reduce_mean_magsq)
    d_reduce_accum_ss(space, ncols, 1, nsteps, radix, tid);
  else if (type == reduce_min_magsq || type == reduce_min_mag)
    d_reduce_accum_min_ss(space, space_idx, ncols, 1, nsteps, radix, tid);

  if (tid == 0)
  {
    if (type == reduce_max_magsq || type == reduce_max_mag)
    {
      sum[0] = space[0];
      *row_idx = 0;
      *col_idx = space_idx[0];
    }
    else if (type == reduce_mean_magsq)
      sum[0] += space[0] / (float)nrows;
    else if (type == reduce_min_magsq || type == reduce_min_mag)
    {
      sum[0] = space[0];
      *row_idx = 0;
      *col_idx = space_idx[0];
    }
  }

  __syncthreads();

  for (int i = 1; i < nrows; ++i)
  {
    d_reduce_magsq_copy_cs(input + __mul24(ncols, i), space, ncols, radix, tid, type);

    __syncthreads();

    if (type == reduce_max_magsq || type == reduce_max_mag)
      d_reduce_accum_max_ss(space, space_idx, ncols, 1, nsteps, radix, tid);
    else if (type == reduce_mean_magsq)
      d_reduce_accum_ss(space, ncols, 1, nsteps, radix, tid);
    else if (type == reduce_min_magsq || type == reduce_min_mag)
      d_reduce_accum_min_ss(space, space_idx, ncols, 1, nsteps, radix, tid);

    if (tid == 0)
    {
      if (type == reduce_max_magsq || type == reduce_max_mag)
      {
        if (space[0] > sum[0])
        {
          sum[0] = space[0];
          *row_idx = i;
          *col_idx = space_idx[0];
        }
      }
      else if (type == reduce_mean_magsq)
        sum[0] += space[0] / (float)nrows;
      else if (type == reduce_min_magsq || type == reduce_min_mag)
      {
        if (space[0] < sum[0])
        {
          sum[0] = space[0];
          *row_idx = i;
          *col_idx = space_idx[0];
        }
      }
    }
    __syncthreads();
  }
}

namespace vsip
{
namespace impl
{
namespace cuda
{


template <typename T, typename R>
R
reduce(T const* input,
       size_t   numrows,
       size_t   numcols,
       size_t   &row_idx,
       size_t   &col_idx,
       int      reduction_type);

template <>
float
reduce<float, float>(float const* input,
                     size_t       numrows,
                     size_t       numcols,
                     size_t       &row_idx,
                     size_t       &col_idx,
                     int          reduction_type)
{
   // Device temporary storage for intermediate results.
  float *buffer, *out, output;
  size_t nele, stages, *buffer_idx, *out_row_idx, *out_col_idx;
  size_t threads = Dev_props::max_threads_per_block();

 
  // Number of elements necessary to keep required
  //  threads <= max_threads_per_block()
  nele = (numcols + threads - 1) / threads + 1;

  // Number of stages, this will be more than required for chunks > 2.
  stages = ceil(log2(float(numcols)));
  
  cudaMalloc((void**)&out, sizeof(float));
  cudaMalloc((void**)&buffer, numcols * sizeof(float));
  cudaMalloc((void**)&buffer_idx, numcols * sizeof(size_t));
  cudaMalloc((void**)&out_row_idx, sizeof(size_t));
  cudaMalloc((void**)&out_col_idx, sizeof(size_t));

  k_reduce_ss<<<1, threads>>>(input, buffer, out, buffer_idx, out_row_idx, out_col_idx,
                              numrows, numcols, stages, nele, (reduce_type)reduction_type);


  cudaMemcpy(&output, out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&col_idx, out_col_idx, sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&row_idx, out_row_idx, sizeof(size_t), cudaMemcpyDeviceToHost);

  cudaFree(out);
  cudaFree(buffer);
  cudaFree(buffer_idx);
  cudaFree(out_row_idx);
  cudaFree(out_col_idx);

  return(output);

}

template <>
std::complex<float>
reduce<std::complex<float>, std::complex<float> >(std::complex<float> const* input,
                                                  size_t                     numrows,
                                                  size_t                     numcols,
                                                  size_t                     &row_idx,
                                                  size_t                     &col_idx,
                                                  int                        reduction_type)
{
  // Device temporary storage for intermediate results.
  std::complex<float> *buffer, *out, output;
  int nele, stages;
  size_t threads = Dev_props::max_threads_per_block();

  // Number of elements necessary to keep required
  //  threads <= max_threads_per_block()
  nele = (numcols + threads - 1) / threads + 1;

  // Number of stages, this will be more than required for chunks > 2.
  stages = ceil(log2(float(numcols)));
  
  cudaMalloc((void**)&out, sizeof(std::complex<float>));
  cudaMalloc((void**)&buffer, numcols * sizeof(std::complex<float>));

  k_reduce_cc<<<1, threads>>>(reinterpret_cast<cuComplex const*>(input),
                              reinterpret_cast<cuComplex*>(buffer),
                              reinterpret_cast<cuComplex*>(out), numrows,
                              numcols, stages, nele, (reduce_type)reduction_type);


  cudaMemcpy(&output, out, sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

  cudaFree(out);
  cudaFree(buffer);

  return(output);
}

template <>
float
reduce<std::complex<float>, float>(std::complex<float> const* input,
                                   size_t                     numrows,
                                   size_t                     numcols,
                                   size_t                     &row_idx,
                                   size_t                     &col_idx,
                                   int                        reduction_type)
{
  // Device temporary storage for intermediate results.
  float *buffer, *out, output;
  int nele, stages;
  size_t threads = Dev_props::max_threads_per_block();
  size_t *buffer_idx, *out_row_idx, *out_col_idx;

  // Number of elements necessary to keep required
  //  threads <= max_threads_per_block()
  nele = (numcols + threads - 1) / threads + 1;

  // Number of stages, this will be more than required for chunks > 2.
  stages = ceil(log2(float(numcols)));
  
  cudaMalloc((void**)&out, sizeof(float));
  cudaMalloc((void**)&buffer, numcols * sizeof(float));
  cudaMalloc((void**)&buffer_idx, numcols * sizeof(size_t));
  cudaMalloc((void**)&out_row_idx, sizeof(size_t));
  cudaMalloc((void**)&out_col_idx, sizeof(size_t));

  k_reduce_magsq_cs<<<1, threads>>>(reinterpret_cast<cuComplex const*>(input),
                              buffer, out, buffer_idx, out_row_idx, out_col_idx,
                              numrows, numcols, stages, nele, (reduce_type)reduction_type);


  cudaMemcpy(&output, out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&col_idx, out_col_idx, sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&row_idx, out_row_idx, sizeof(size_t), cudaMemcpyDeviceToHost);

  cudaFree(out);
  cudaFree(buffer);
  cudaFree(buffer_idx);
  cudaFree(out_row_idx);
  cudaFree(out_col_idx);

  
  return(output);
}

} // vsip::impl::cuda
} // vsip::impl
} // vsip

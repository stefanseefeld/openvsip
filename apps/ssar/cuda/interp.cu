/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   User-defined polar to rectangular interpolation kernel for SSAR images.

#include <vsip/support.hpp>
#include <cuComplex.h>
#include <complex>

// Shared memory pool, allocated dynamically at kernel launch time.  This 
// is allocated out of the multiprocessor's physical shared memory.  Multiple 
// blocks may be allocated from physical memory, each will get their own 
// pool.  All threads within the block see the same shared memory.
extern __shared__ char shared[];


/***********************************************************************
  Device functions (callable only via kernels)
***********************************************************************/

// scalar-complex multiply
//   c = a * b   where b and c are complex and a is real
__device__ void scmul(cuComplex& c, float a, cuComplex b)
{
  c.x = a * b.x;
  c.y = a * b.y;
}

// scalar-complex multiply-add
//   c += a * b   where b and c are complex and a is real
__device__ void scmadd(cuComplex& c, float a, cuComplex b)
{
  c.x += a * b.x;
  c.y += a * b.y;
}


/***********************************************************************
  Device Kernels
***********************************************************************/

__global__ void 
k_zerofill_c(cuComplex* inout, size_t size)
{
  int const idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

  if (idx < size)
  {
    inout[idx].x = 0.f;
    inout[idx].y = 0.f;
  }
}


__global__ void 
k_interpolate(
  unsigned int const* indices,      // m x n
  float const*        window,       // m x n x I
  cuComplex const*    in,           // m x n
  cuComplex*          out,          // m x nx
  size_t              depth,        // I
  size_t              wstride,      // I + pad bytes
  size_t              rows,         // m
  size_t              cols_in,      // n
  size_t              cols_out,     // nx
  size_t              subcols,      // number of sub-columns handled by a thread
  bool                shift_input)  // swap left and right halves of input
{
  // Rows are distributed by block and columns are distributed in groups
  // spaced a fixed amount apart.
  int const row = blockIdx.x;
  int const col = threadIdx.x * subcols;

  // Set pointers to the correct row for this thread
  indices += row * cols_in;
  window += row * cols_in * wstride;
  in += row * cols_in;
  out += row * cols_out;

  // Set up a pointer to this thread's shared memory buffer and
  // clear the memory
  cuComplex* k_accum = (cuComplex*) shared;
  k_accum += __mul24(threadIdx.x, subcols + depth - 1);
  for (int i = 0; i < subcols + depth - 1; ++i)
  {
    k_accum[i].x = 0;
    k_accum[i].y = 0;
  }


  // Ensure the start element in this sub-column is within range of the image
  if (col < cols_in)
  {
    if (subcols > (cols_in - col))
      subcols = cols_in - col;

    // All threads in this warp will step through these loops together
    // in lock-step.  As a result, the synchronization is implicit and
    // no __syncthreads() calls are needed.
    for (int i = col; i < col + subcols; ++i)
    {
      int i_shift = shift_input ? (i + cols_in/2) % cols_in : i;
      float const* pw = window + __mul24(i, wstride);
      cuComplex const input = in[i_shift];
      int const ikxcols = indices[i];

      // Each input is multiplied by the window vector and the result
      // accumulated in the correct position in the shared memory buffer.
      int ikx = ikxcols - indices[col];
      for (int h = 0; h < depth; ++h)
        scmadd(k_accum[ikx + h], pw[h], input);
    }


    // The shared memory values are now merged with the output values in
    // global memory.
    int ikx0 = indices[col];
    for (int i = 0; i < subcols + depth - 1; ++i)	
    {
      out[ikx0 + i].x += k_accum[i].x;
      out[ikx0 + i].y += k_accum[i].y;
    }
  }
}


__global__ void 
k_freq_domain_fftshift(cuComplex const* input, cuComplex* output, size_t rows, size_t cols)
{
  int const row = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
  int const col = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

  // Ensure this element is within bounds of the image
  if ((row < rows) && (col < cols))
  { 
    int const idx = __mul24(row, cols) + col;

    if ((row & 1) == (col & 1))
      scmul(output[idx], -1, input[idx]);
  }
}

namespace cuda
{

inline
size_t
compute_shared_memory_size(size_t subblocks, size_t subcols, size_t depth)
{
  // calculate amount of shared memory (in bytes) needed.
  return 
    subblocks * (subcols + depth - 1) * sizeof(cuComplex);  // accumulation buffer
}

inline 
void
distribute_vector(size_t size, dim3& grid, dim3& threads)
{
  threads.x = 512;
  grid.x = (size + threads.x - 1) / threads.x;
}

inline 
void
distribute_matrix(size_t rows, size_t cols, dim3& grid, dim3& threads)
{
  threads.y = 16;
  threads.x = 32;

  grid.y = (rows + threads.y - 1) / threads.y;
  grid.x = (cols + threads.x - 1) / threads.x;
}

inline
void
distribute_by_row_and_subcol(size_t cols_in, size_t& subblocks, size_t& subcols)
{
  subblocks = 32;
  subcols = (cols_in + subblocks - 1) / subblocks;
  while (subcols < subblocks)
  { 
    subcols *= 2;
    subblocks /= 2;
  }
}



void
interpolate(unsigned int const *indices,  // m x n
	    float const *window,          // m x n x I
	    std::complex<float> const *in,// m x n
	    std::complex<float> *out,     // m x nx
	    vsip::length_type depth,      // I
	    vsip::length_type wstride,    // I + pad bytes (if any)
	    vsip::length_type rows,       // m
	    vsip::length_type cols_in,    // n
	    vsip::length_type cols_out)   // nx
{
  // Clear the output buffer in it's entirety.  because it is 
  // contiguous, it may be treated as linear memory.
  {
    dim3 grid, threads;
    distribute_vector(rows * cols_out, grid, threads);
    k_zerofill_c<<<grid, threads>>>(reinterpret_cast<cuComplex*>(out), rows * cols_out);
    cudaThreadSynchronize();
  }

  // Compute the results
  {
    size_t subblocks;
    size_t subcols;
    distribute_by_row_and_subcol(cols_in, subblocks, subcols);

    // This determines the amount of shared memory to be dynamically
    // allocated when the kernel is launched.  The kernel's shared
    // memory requirements are implicit, so care must be taken to 
    // provide a sufficient amount.
    size_t sm_size = compute_shared_memory_size(subblocks, subcols, depth);

    // Each block processes one entire row (line) of the image.
    // The number of sub-columns in a group tells each thread how 
    // many elements to process.  The flag set to false tells it
    // not to FFT-shift the input (swap left and right halves).
    dim3 grid(rows);
    dim3 threads(subblocks);
    k_interpolate<<<grid, threads, sm_size>>>
      (indices, window,
       reinterpret_cast<cuComplex const*>(in),
       reinterpret_cast<cuComplex *>(out), 
       depth, wstride, rows, cols_in, cols_out, subcols, false);
    cudaThreadSynchronize();
  }
}

void
interpolate_with_shift(unsigned int const *indices,  // m x n
		       float const *window,          // m x n x I
		       std::complex<float> const *in,// m x n
		       std::complex<float> *out,     // m x nx
		       vsip::length_type depth,      // I
		       vsip::length_type wstride,    // I + pad bytes (if any)
		       vsip::length_type rows,       // m
		       vsip::length_type cols_in,    // n
		       vsip::length_type cols_out)   // nx
{
  // Clear the output buffer in it's entirety.  because it is 
  // contiguous, it may be treated as linear memory.
  {
    dim3 grid, threads;
    distribute_vector(rows * cols_out, grid, threads);
    k_zerofill_c<<<grid, threads>>>(reinterpret_cast<cuComplex*>(out), rows * cols_out);
    cudaThreadSynchronize();
  }

  // Compute the results
  {
    size_t subblocks;
    size_t subcols;
    distribute_by_row_and_subcol(cols_in, subblocks, subcols);

    // This determines the amount of shared memory to be dynamically
    // allocated when the kernel is launched.  The kernel's shared
    // memory requirements are implicit, so care must be taken to 
    // provide a sufficient amount.
    size_t sm_size = compute_shared_memory_size(subblocks, subcols, depth);

    // Each block processes one entire row (line) of the image.
    // The number of sub-columns in a group tells each thread how 
    // many elements to process.  The flag set to false tells it
    // not to FFT-shift the input (swap left and right halves).
    dim3 grid(rows);
    dim3 threads(subblocks);
    k_interpolate<<<grid, threads, sm_size>>>
      (indices, window,
       reinterpret_cast<cuComplex const*>(in),
       reinterpret_cast<cuComplex *>(out), 
       depth, wstride, rows, cols_in, cols_out, subcols, true);
    cudaThreadSynchronize();
  }


  // Perform a second fftshift, but in the frequency domain.  Combining
  // this kernel with the above results in a kernel that is too large
  // to be launched.  Keeping this part separate has the added advantage
  // of being able to distribute one element per thread instead of one
  // row per thread.
  {
    dim3 grid, threads;
    distribute_matrix(rows, cols_out, grid, threads);
    k_freq_domain_fftshift<<<grid, threads>>>
      (reinterpret_cast<cuComplex*>(out),
       reinterpret_cast<cuComplex*>(out),
       rows, cols_out);
    cudaThreadSynchronize();
  }
}

} // namespace cuda

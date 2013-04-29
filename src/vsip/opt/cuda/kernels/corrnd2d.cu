/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA kernels for optimized 2D convolution.
#include <cuComplex.h>
#include <complex>
#include <cmath>
#include "cmplx.cuh"
#include "util.hpp"

using namespace dev;

//  The CUDA computation of the 2-D correlation uses multiple stages in which
// the results for different "tiles" are computed.  The entire computation is
// separated into tiles representing:
//  a: the center region for which there is no zero padding required
//  b: the corner regions in which zero padding is required in both dimensions
//  c: the edges along the first dimension in which zero padding is required
//     along the row but not along the columns
//  d: the edges along the second dimension in which zero padding is required
//     along the column but not along the row.

//  Unbiasing is done within each tile region.

// Type for the tile region
enum tile_region_type
{
  center,
  corners,
  left_right_edge,
  top_bottom_edge
};

// Type for the support type including the even-ness of the kernel size
enum supp_region_type
{
  full_supprt,
  same_supprt_nrows_even_ncols_even,
  same_supprt_nrows_odd_ncols_even,
  same_supprt_nrows_even_ncols_odd,
  same_supprt_nrows_odd_ncols_odd,
  min_supprt
};

// Shared memory amount to be determined at runtime
extern __shared__ char shared_array_s[];

// Device function to perform the copy of data from global memory to shared
//  memory depending on "tile".
template<typename T>
__device__ inline void
d_load_global_to_shared(
  T const* input,
  T const* kernel,
  T*       input_sh,
  T*       kernel_sh,
  int      kernel_nrows,
  int      kernel_ncols,
  int      input_nrows,
  int      input_ncols,
  tile_region_type tile)
{
  int tr = threadIdx.x;
  int tc = threadIdx.y;
  int tidr = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
  int tidc = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
  int width = __mul24(2, blockDim.y);

  if (tile == center)
  {
    if (tr < kernel_nrows && tc < kernel_ncols)
      kernel_sh[__mul24(tr, blockDim.y) + tc] =
         kernel[__mul24(tr, kernel_ncols) + tc];

    if (tidr < input_nrows && tidc < input_ncols)
      input_sh[__mul24(tr, width) + tc] =
         input[__mul24(tidr, input_ncols) + tidc];

    if (tidr + blockDim.x < input_nrows && tidc < input_ncols)
      input_sh[__mul24(tr + blockDim.x, width) + tc] =
         input[__mul24(tidr + blockDim.x, input_ncols) + tidc];

    if (tidr < input_nrows && tidc + blockDim.y < input_ncols)
      input_sh[__mul24(tr, width) + tc + blockDim.y] =
         input[__mul24(tidr, input_ncols) + tidc + blockDim.y];

    if (tidr + blockDim.x < input_nrows && tidc + blockDim.y < input_ncols)
      input_sh[__mul24(tr + blockDim.x, width) + tc + blockDim.y] =
         input[__mul24(tidr + blockDim.x, input_ncols) + tidc + blockDim.y];
  }
  else if (tile == corners)
  {
    kernel_sh[__mul24(tr, blockDim.y) + tc] =
       kernel[__mul24(tr, kernel_ncols) + tc];

    input_sh[__mul24(tr, width) + tc] = input[__mul24(tr, input_ncols) + tc];

    input_sh[__mul24(tr + kernel_nrows, width) + tc] =
       input[__mul24(tr + input_nrows - kernel_nrows, input_ncols) + tc];

    input_sh[__mul24(tr, width) + tc + kernel_ncols] =
       input[__mul24(tr, input_ncols) + tc + input_ncols - kernel_ncols];

    input_sh[__mul24(tr + kernel_nrows, width) + tc + kernel_ncols] =
       input[__mul24(tr + input_nrows - kernel_nrows, input_ncols) +
             tc + input_ncols - kernel_ncols];
  }
  else if (tile == left_right_edge)
  {
    if (tr < kernel_nrows)
      kernel_sh[__mul24(tr, blockDim.y) + tc] =
         kernel[__mul24(tr, kernel_ncols) + tc];

    if (tidr < input_nrows)
    {
      input_sh[__mul24(tr, width) + tc] =
         input[__mul24(tidr, input_ncols) + tc];

      input_sh[__mul24(tr, width) + tc + kernel_ncols] =
         input[__mul24(tidr, input_ncols) + tc + input_ncols - kernel_ncols];
    }

    if (tidr + blockDim.x < input_nrows)
    {
      input_sh[__mul24(tr + blockDim.x, width) + tc] =
         input[__mul24(tidr + blockDim.x, input_ncols) + tc];

      input_sh[__mul24(tr + blockDim.x, width) + tc + kernel_ncols] =
         input[__mul24(tidr + blockDim.x, input_ncols) +
               tc + input_ncols - kernel_ncols];
    }
  }
  else if (tile == top_bottom_edge)
  {
    if (tc < kernel_ncols)
      kernel_sh[__mul24(tr, blockDim.y) + tc] =
         kernel[__mul24(tr, kernel_ncols) + tc];

    if (tidc < input_ncols)
    {
      input_sh[__mul24(tr, width) + tc] =
         input[__mul24(tr, input_ncols) + tidc];

      input_sh[__mul24(tr + kernel_nrows, width) + tc] =
         input[__mul24(tr + input_nrows - kernel_nrows, input_ncols) + tidc];
    }

    if (tidc + blockDim.y < input_ncols)
    {
      input_sh[__mul24(tr, width) + tc + blockDim.y] =
         input[__mul24(tr, input_ncols) + tidc + blockDim.y];

      input_sh[__mul24(tr + kernel_nrows, width) + tc + blockDim.y] =
      input[__mul24(tr + input_nrows - kernel_nrows, input_ncols) +
            tidc + blockDim.y];
    }
  }
}

template void d_load_global_to_shared<float>(
                  float const*, float const*, float*, float*, int, int, int,
                  int, tile_region_type);

template void d_load_global_to_shared<cuComplex>(
                  cuComplex const*, cuComplex const*, cuComplex*, cuComplex*,
                  int, int, int, int, tile_region_type);

//  Device function to store a value or set of values ("sum_x") to the
//   appropriate global memory region depending on "tile" and "supp"
template<typename T>
__device__ inline void
d_store_var_to_global(
  T*       out,
  T        sum,
  T        sum_begin,
  T        sum_end,
  T        sum_ur,
  T        sum_ul,
  T        sum_ll,
  T        sum_lr,
  int      kernel_nrows,
  int      kernel_ncols,
  int      input_nrows,
  int      input_ncols,
  int      output_nrows,
  int      output_ncols,
  int      row_shift,
  int      col_shift,
  tile_region_type tile,
  supp_region_type supp)
{
  int tr = threadIdx.x;
  int tc = threadIdx.y;
  int tidr = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
  int tidc = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

  if (tile == center)
  {
    if (supp == full_supprt)
    {
      if (tidc < input_ncols - kernel_ncols + 1 &&
          tidr < input_nrows - kernel_nrows + 1)
        *(out + __mul24(tidr + kernel_nrows - 1, output_ncols) +
                tidc + kernel_ncols - 1) = sum;
    }
    else if (supp == same_supprt_nrows_even_ncols_even)
    {
      if (tidc < input_ncols - kernel_ncols + 1 &&
          tidr < input_nrows - kernel_nrows + 1)
        *(out + __mul24(tidr - row_shift, output_ncols) +
                tidc - col_shift) = sum;
    }
    else if (supp == same_supprt_nrows_even_ncols_odd)
    {
      if (tidc < input_ncols - kernel_ncols + 1 &&
          tidr < input_nrows - kernel_nrows + 1)
        *(out + __mul24(tidr - row_shift, output_ncols) +
                tidc - col_shift) = sum;
    }
    else if (supp == same_supprt_nrows_odd_ncols_even)
    {
      if (tidc < input_ncols - kernel_ncols + 1 &&
          tidr < input_nrows - kernel_nrows + 1)
        *(out + __mul24(tidr - row_shift, output_ncols) +
                tidc - col_shift) = sum;
    }
    else if (supp == same_supprt_nrows_odd_ncols_odd)
    {
      if (tidc < input_ncols - kernel_ncols + 1 &&
          tidr < input_nrows - kernel_nrows + 1)
        *(out + __mul24(tidr - row_shift, output_ncols) +
                tidc - col_shift) = sum;
    }
    else if (supp == min_supprt)
    {
      if (tidc < output_ncols && tidr < output_nrows)
        *(out + __mul24(tidr, output_ncols) + tidc) = sum;
    }
  }
  else if (tile == corners)
  {
    if (supp == full_supprt)
    {
      if (tr < kernel_nrows - 1 && tc < kernel_ncols - 1)
      {
        out[__mul24(tr, output_ncols) + tc] = sum_ul;
        out[__mul24(output_nrows - tr - 1, output_ncols) + tc] = sum_ll;
        out[__mul24(tr, output_ncols) + output_ncols - tc - 1] = sum_ur;
        out[__mul24(output_nrows - tr - 1, output_ncols) +
            output_ncols - tc - 1] = sum_lr;
      }
    }
    else if (supp == same_supprt_nrows_even_ncols_even)
    {
      if (tr == 0 && tc == 0)
        out[__mul24(tr, output_ncols) +
            tc] = sum_ul;
      else if (tr == 0 && tc < kernel_ncols + col_shift)
      {
        out[__mul24(tr, output_ncols) + tc] = sum_ul;
        out[__mul24(tr, output_ncols) +
            output_ncols - tc] = sum_ur;
      }
      else if (tr < kernel_nrows + row_shift && tc == 0)
      {
        out[__mul24(tr, output_ncols) + tc] = sum_ul;
        out[__mul24(output_nrows - tr, output_ncols) +
            tc] = sum_ll;
      }
      else if (tr < kernel_nrows + row_shift && tr > 0 &&
               tc < kernel_ncols + col_shift && tc > 0)
      {
        out[__mul24(tr, output_ncols) + tc] = sum_ul;
        out[__mul24(output_nrows - tr, output_ncols) + tc] = sum_ll;
        out[__mul24(tr, output_ncols) + output_ncols - tc] = sum_ur;
        out[__mul24(output_nrows - tr, output_ncols) +
            output_ncols - tc] = sum_lr;
      }
    }
    else if (supp == same_supprt_nrows_even_ncols_odd)
    {
      if (tc < kernel_ncols + col_shift - 1)
      {
        if (tr == 0)
        {
          out[__mul24(tr, output_ncols) + tc] = sum_ul;
          out[__mul24(tr, output_ncols) +
              output_ncols - tc - 1] = sum_ur;
        }
        else if (tr < kernel_nrows + row_shift)
        {
          out[__mul24(tr, output_ncols) + tc] = sum_ul;
          out[__mul24(output_nrows - tr, output_ncols) + tc] = sum_ll;
          out[__mul24(tr, output_ncols) + output_ncols - tc - 1] = sum_ur;
          out[__mul24(output_nrows - tr, output_ncols) +
              output_ncols - tc - 1] = sum_lr;
        }
      }
    }
    else if (supp == same_supprt_nrows_odd_ncols_even)
    {
      if (tr < kernel_nrows + row_shift - 1)
      {
        if (tc == 0)
        {
          out[__mul24(tr, output_ncols) + tc] = sum_ul;
          out[__mul24(output_nrows - tr - 1, output_ncols) +
              tc] = sum_ll;
        }
        else if (tc < kernel_ncols + col_shift)
        {
          out[__mul24(tr, output_ncols) + tc] = sum_ul;
          out[__mul24(output_nrows - tr - 1, output_ncols) + tc] = sum_ll;
          out[__mul24(tr, output_ncols) + output_ncols - tc] = sum_ur;
          out[__mul24(output_nrows - tr - 1, output_ncols) +
              output_ncols - tc] = sum_lr;
        }
      }
    }
    else if (supp == same_supprt_nrows_odd_ncols_odd)
    {
      if (tr < kernel_nrows + row_shift - 1 &&
          tc < kernel_ncols + col_shift - 1)
      {
        out[__mul24(tr, output_ncols) + tc] = sum_ul;
        out[__mul24(output_nrows - tr - 1, output_ncols) + tc] = sum_ll;
        out[__mul24(tr, output_ncols) + output_ncols - tc - 1] = sum_ur;
        out[__mul24(output_nrows - tr - 1, output_ncols) +
            output_ncols - tc - 1] = sum_lr;
      }
    }
  }
  else if (tile == left_right_edge)
  {
    if (supp == full_supprt)
    {
      if (tidr < input_nrows - kernel_nrows + 1 && tc < kernel_ncols - 1)
      {
        out[__mul24(tidr + kernel_nrows - 1, output_ncols) + tc] = sum_begin;
        out[__mul24(tidr + kernel_nrows - 1, output_ncols) +
            output_ncols - tc - 1] = sum_end;
      }
    }
    else if (supp == same_supprt_nrows_even_ncols_even)
    {
      if (tidr < input_nrows - kernel_nrows + 1)
      {
        if (tc == 0)
          out[__mul24(tidr - row_shift, output_ncols) +
              tc] = sum_begin;
        else if (tc < kernel_ncols + col_shift && tc > 0)
        {
          out[__mul24(tidr - row_shift, output_ncols) + tc] = sum_begin;
          out[__mul24(tidr - row_shift, output_ncols) +
              output_ncols - tc] = sum_end;
        }
      }
    }
    else if (supp == same_supprt_nrows_even_ncols_odd)
    {
      if (tidr < input_nrows - kernel_nrows + 1 &&
            tc < kernel_ncols + col_shift - 1)
      {
        out[__mul24(tidr - row_shift, output_ncols) + tc] = sum_begin;
        out[__mul24(tidr - row_shift, output_ncols) +
                    output_ncols - tc - 1] = sum_end;
      }
    }
    else if (supp == same_supprt_nrows_odd_ncols_even)
    {
      if (tidr < input_nrows - kernel_nrows + 1)
      {
        if (tc == 0)
          out[__mul24(tidr - row_shift, output_ncols) +
              tc] = sum_begin;
        else if (tc < kernel_ncols + col_shift)
        {
          out[__mul24(tidr - row_shift, output_ncols) + tc] = sum_begin;
          out[__mul24(tidr - row_shift, output_ncols) +
              output_ncols - tc] = sum_end;
        }
      }
    }
    else if (supp == same_supprt_nrows_odd_ncols_odd)
    {
      if (tidr < input_nrows - kernel_nrows + 1 &&
            tc < kernel_ncols + col_shift - 1)
      {
        out[__mul24(tidr - row_shift, output_ncols) + tc] = sum_begin;
        out[__mul24(tidr - row_shift, output_ncols) +
            output_ncols - tc - 1] = sum_end;
      }
    }
  }
  else if (tile == top_bottom_edge)
  {
    if (supp == full_supprt)
    {
      if (tr < kernel_nrows - 1 && tidc < input_ncols - kernel_ncols + 1)
      {
        out[__mul24(tr, output_ncols) + tidc + kernel_ncols - 1] = sum_begin;
        out[__mul24(output_nrows - tr - 1, output_ncols) +
            tidc + kernel_ncols - 1] = sum_end;
      }
    }
    else if (supp == same_supprt_nrows_even_ncols_even)
    {
      if (tidc < input_ncols - kernel_ncols + 1)
      {
        if (tr == 0)
          out[__mul24(tr, output_ncols) +
              tidc - col_shift] = sum_begin;
        else if (tr < kernel_nrows + row_shift && tr > 0)
        {
          out[__mul24(tr, output_ncols) + tidc - col_shift] = sum_begin;

          out[__mul24(output_nrows - tr, output_ncols) +
              tidc - col_shift] = sum_end;
        }
      }
    }
    else if (supp == same_supprt_nrows_even_ncols_odd)
    {
      if (tidc < input_ncols - kernel_ncols + 1)
      {
        if (tr == 0)
          out[__mul24(tr, output_ncols) +
              tidc - col_shift] = sum_begin;
        else if (tr < kernel_nrows + row_shift)
        {
          out[__mul24(tr, output_ncols) + tidc - col_shift] = sum_begin;
          out[__mul24(output_nrows - tr, output_ncols) +
              tidc - col_shift] = sum_end;
        }
      }
    }
    else if (supp == same_supprt_nrows_odd_ncols_even)
    {
      if (tidc < input_ncols - kernel_ncols + 1 &&
            tr < kernel_nrows - 1 + row_shift)
      {
        out[__mul24(tr, output_ncols) + tidc - col_shift] = sum_begin;
        out[__mul24(output_nrows - tr - 1, output_ncols) +
            tidc - col_shift] = sum_end;
      }
    }
    else if (supp == same_supprt_nrows_odd_ncols_odd)
    {
      if (tidc < input_ncols - kernel_ncols + 1 &&
            tr < kernel_nrows + row_shift - 1)
      {
        out[__mul24(tr, output_ncols) + tidc - col_shift] = sum_begin;
        out[__mul24(output_nrows - tr - 1, output_ncols) +
            tidc - col_shift] = sum_end;
      }
    }
  }
}

template void d_store_var_to_global<float>(float*, float, float, float, float,
                                           float,  float, float, int, int, int,
                                           int, int, int, int, int,
                                           tile_region_type, supp_region_type);

template void d_store_var_to_global<cuComplex>(cuComplex*, cuComplex, cuComplex,
                                               cuComplex,  cuComplex, cuComplex,
                                               cuComplex,  cuComplex, int, int,
                                               int, int, int, int, int, int,
                                               tile_region_type, supp_region_type);


// Device function to perform the calculation by looping over rows and columns.
//  The result(s) are stored in the "sum" variables.
template<typename T>
__device__ inline void
d_loops(
  T*       input_sh,
  T*       kernel_sh,
  T&        sum,
  T&        sum_begin,
  T&        sum_end,
  T&        sum_ur,
  T&        sum_ul,
  T&        sum_ll,
  T&        sum_lr,
  int      kernel_nrows,
  int      kernel_ncols,
  int      input_nrows,
  int      input_ncols,
  int      output_nrows,
  int      output_ncols,
  int      row_shift,
  int      col_shift,
  int      delta_tr,
  int      delta_tc,
  int      is_unbiased,
  tile_region_type tile,
  supp_region_type supp)
{
  int tr = threadIdx.x;
  int tc = threadIdx.y;
  int tidr = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
  int tidc = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
  int width = __mul24(2, blockDim.y);
  T temp;

  if (tile == center)
  {
    for (int i = 0; i < kernel_nrows; ++i)
    {
      T sum_row;

      sum_row *= 0.0F;
      for (int j = 0; j < kernel_ncols; ++j)
      {
        int row_index = tr + i;
        int col_index = tc + j;

        cconj(temp, input_sh[__mul24(row_index, width) + col_index]);

        sum_row += temp * kernel_sh[__mul24(i, blockDim.y) + j];
      }
      sum += sum_row;
    }

    if (is_unbiased)
      sum /= float(kernel_nrows * kernel_ncols);
  }
  else if (tile == corners)
  {
    for (int i = 0; i <= delta_tr; ++i)
    {
      T sum_row_ul, sum_row_ur, sum_row_lr, sum_row_ll;

      sum_row_ul *= 0.0;
      sum_row_ur *= 0.0;
      sum_row_lr *= 0.0;
      sum_row_ll *= 0.0;  

      for (int j = 0; j <= delta_tc; ++j)
      {
        cconj(temp, input_sh[__mul24(i, width) + j]);

        sum_row_ul += temp * kernel_sh[__mul24(i - row_shift - tr, blockDim.y) + j - col_shift - tc];

        cconj(temp, input_sh[__mul24(i, width) + __mul24(kernel_ncols, 2) - delta_tc + j - 1]);

        sum_row_ur += temp * kernel_sh[__mul24(i - row_shift - tr, blockDim.y) + j];

        cconj(temp, input_sh[(__mul24(kernel_nrows, 2) - delta_tr + i - 1) * width + j]);

        sum_row_ll += temp * kernel_sh[__mul24(i, blockDim.y) + j - col_shift - tc];

        cconj(temp, input_sh[(__mul24(kernel_nrows, 2) - delta_tr + i - 1) * width + __mul24(kernel_ncols, 2) - delta_tc + j - 1]);

        sum_row_lr += temp * kernel_sh[__mul24(i, blockDim.y) + j];
      }
      sum_ul += sum_row_ul;
      sum_ur += sum_row_ur;
      sum_ll += sum_row_ll;
      sum_lr += sum_row_lr;
    }
 
    if (is_unbiased)
    {
      sum_ul /= float((tr + 1) * (tc + 1));
      sum_ur /= float((tr + 1) * (tc + 1));
      sum_ll /= float((tr + 1) * (tc + 1));
      sum_lr /= float((tr + 1) * (tc + 1));
    }
  }
  else if (tile == left_right_edge)
  {
    for (int i = 0; i < kernel_nrows; ++i)
    {
      T sum_row_begin, sum_row_end;

      sum_row_begin *= 0.0;
      sum_row_end *= 0.0;

      for (int j = 0; j <= delta_tc; ++j)
      {
        cconj(temp, input_sh[__mul24(tr + i, width) + j]);

        sum_row_begin += temp * kernel_sh[__mul24(i, blockDim.y) + j - col_shift - tc];

        cconj(temp, input_sh[__mul24(tr + i, width) + __mul24(kernel_ncols, 2) - delta_tc + j - 1]);

        sum_row_end += temp * kernel_sh[__mul24(i, blockDim.y) + j];
      }
      sum_begin += sum_row_begin;
      sum_end += sum_row_end;
    }

    if (is_unbiased)
    {
      sum_begin /= float(kernel_nrows * (tc + 1));
      sum_end /= float(kernel_nrows * (tc + 1));
    }
  }
  else if (tile == top_bottom_edge)
  {
    for (int j = 0; j < kernel_ncols; ++j)
    {
      T sum_col_begin, sum_col_end;

      sum_col_begin *= 0.0;
      sum_col_end *= 0.0;

      for (int i = 0; i <= delta_tr; ++i)
      {
        cconj(temp, input_sh[__mul24(i, width) + tc + j]);

        sum_col_begin += temp * kernel_sh[__mul24(i - row_shift - tr, blockDim.y) + j];

        cconj(temp, input_sh[(__mul24(kernel_nrows, 2) - delta_tr + i - 1) * width + tc + j]);

        sum_col_end += temp * kernel_sh[__mul24(i, blockDim.y) + j];
      }
      sum_begin += sum_col_begin;
      sum_end += sum_col_end;
    }

    if (is_unbiased)
    {
      sum_begin /= float(kernel_ncols * (tr + 1));
      sum_end /= float(kernel_ncols * (tr + 1));
    }
  }
}

template void d_loops<float>(float*, float*, float&, float&, float&,
                             float&,  float&, float&, float&, int, int, int,
                             int, int, int, int, int, int, int, int,
                             tile_region_type, supp_region_type);

template void d_loops<cuComplex>(cuComplex*, cuComplex*, cuComplex&,
                                 cuComplex&,  cuComplex&, cuComplex&,
                                 cuComplex&,  cuComplex&, cuComplex&, int, int,
                                 int, int, int, int, int, int, int, int, int,
                                 tile_region_type, supp_region_type);

// Global kernel function to perform 2-D convolution
template<typename T>
__global__ void
k_corr2dnd(
  T const* input,
  T const* kernel,
  T*       out,
  int          input_nrows,
  int          input_ncols,
  int          kernel_nrows,
  int          kernel_ncols,
  int          output_nrows,
  int          output_ncols,
  int          row_shift,
  int          col_shift,
  int          del_tr,
  int          del_tc,
  int                 bias,
  tile_region_type tile,
  supp_region_type supp)
{
  int tr = threadIdx.x;
  int tc = threadIdx.y;
  int br = blockIdx.x;
  int bc = blockIdx.y;
  int tidr = __mul24(blockDim.x, br) + tr;
  int tidc = __mul24(blockDim.y, bc) + tc;
  int delta_tr = tr + del_tr;
  int delta_tc = tc + del_tc;

  T sum, sum_begin, sum_end, sum_ur, sum_ul, sum_ll, sum_lr;

  sum *= 0.0;
  sum_begin *= 0.0;
  sum_end *= 0.0;
  sum_ur *= 0.0;
  sum_ul *= 0.0;
  sum_ll *= 0.0;
  sum_lr *= 0.0;

  T *kernel_sh = (T*)shared_array_s;
  T *input_sh = (T*)&shared_array_s[blockDim.x * blockDim.y *
                                    sizeof(T) / sizeof(char)];

  d_load_global_to_shared<T>(input, kernel, input_sh, kernel_sh, kernel_nrows,
                             kernel_ncols, input_nrows, input_ncols, tile);

  __syncthreads();

  d_loops<T>(input_sh, kernel_sh, sum, sum_begin, sum_end, sum_ur, sum_ul,
             sum_ll, sum_lr, kernel_nrows, kernel_ncols, input_nrows,
             input_ncols, output_nrows, output_ncols, row_shift, col_shift,
             delta_tr, delta_tc, bias, tile, supp);

  d_store_var_to_global<T>(out, sum, sum_begin, sum_end, sum_ur, sum_ul,
                           sum_ll, sum_lr, kernel_nrows, kernel_ncols,
                           input_nrows, input_ncols, output_nrows,
                           output_ncols, row_shift, col_shift, tile, supp);
}



namespace vsip
{
namespace impl
{
namespace cuda
{

// Kernels are launched with 16000 bytes of shared memory per block in order
//  to keep enough shared memory to hold the data but to also leave enough
//  room for function arguments, static memory, and execution configuration.
// The number of threads to launch is based on the use of a A x A/2 size
//  array in shared memory for the kernel thus requiring a maximum 2A x A
//  size array for the input.  Thus the required amount of shared memory is
//  (A^2)/2 + 2(A^2) elements = 2.5A^2 elements.
void
corr_2d_no_decimation_min(
  float const*        in,
  float const*        kr,
  float*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads;

  int dtr = 0;
  int dtc = 0;

  size_t const shared_memory_launch_size = 16000;

  distribute_matrix(ou_nc, ou_nr, grid, threads);

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr,
               dtc, bias, center, min_supprt);

}

void
corr_2d_no_decimation_min(
  std::complex<float> const*        in,
  std::complex<float> const*        kr,
  std::complex<float>*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads;

  int dtr = 0;
  int dtc = 0;

  size_t const shared_memory_launch_size = 16000;

  threads.x = min(int(Dev_props::max_threads_per_block_x()),
                  int(sqrt(float(Dev_props::shared_memory_size() - 72) /
                                (8.0 * 2.5))));

  threads.y = threads.x / 2;

  grid.x = (in_nr - kr_nr + threads.x) / threads.x;
  grid.y = (in_nc - kr_nc + threads.y) / threads.y;

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
              reinterpret_cast<cuComplex const*>(in),
              reinterpret_cast<cuComplex const*>(kr),
              reinterpret_cast<cuComplex*>(out), int(in_nr), int(in_nc),
              int(kr_nr), int(kr_nc), int(ou_nr), int(ou_nc), int(rshift),
              int(cshift), dtr, dtc, bias, center, min_supprt);

}

void
corr_2d_no_decimation_full(
  float const*        in,
  float const*        kr,
  float*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads, grid_corner, threads_corner, grid_lr_edge,
       threads_lr_edge, grid_tb_edge, threads_tb_edge;

  int dtr = 0;
  int dtc = 0;

  size_t const shared_memory_launch_size = 16000;

  distribute_matrix(in_nc - kr_nc + 1, in_nr - kr_nr + 1, grid, threads);
  grid_corner.x = 1;
  grid_corner.y = 1;
  threads_corner.x = kr_nr;
  threads_corner.y = kr_nc;

  grid_lr_edge.x = (in_nr - kr_nr + int(Dev_props::max_threads_per_block_x())) /
                                    int(Dev_props::max_threads_per_block_x());
  grid_lr_edge.y = 1;

  grid_tb_edge.x = 1;
  grid_tb_edge.y = (in_nc - kr_nc + int(Dev_props::max_threads_per_block_y())) /
                                    int(Dev_props::max_threads_per_block_y());

  threads_lr_edge.x = int(Dev_props::max_threads_per_block_x());
  threads_lr_edge.y = kr_nc;

  threads_tb_edge.x = kr_nr;
  threads_tb_edge.y = int(Dev_props::max_threads_per_block_y());

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               center, full_supprt);

  k_corr2dnd<<<grid_corner, threads_corner, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               corners, full_supprt);

  k_corr2dnd<<<grid_lr_edge, threads_lr_edge, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               left_right_edge, full_supprt);

  k_corr2dnd<<<grid_tb_edge, threads_tb_edge, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               top_bottom_edge, full_supprt);

}

void
corr_2d_no_decimation_full(
  std::complex<float> const*        in,
  std::complex<float> const*        kr,
  std::complex<float>*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads, grid_corner, threads_corner, grid_lr_edge,
       threads_lr_edge, grid_tb_edge, threads_tb_edge;

  int dtr = 0;
  int dtc = 0;

  size_t const shared_memory_launch_size = 16000;

  threads.x = min(int(Dev_props::max_threads_per_block_x()),
                  int(sqrt(float(Dev_props::shared_memory_size() - 72) /
                                (8.0 * 2.5))));
  threads.y = threads.x / 2;


  grid.x = (in_nr - kr_nr + threads.x) / threads.x;
  grid.y = (in_nc - kr_nc + threads.y) / threads.y;



  grid_corner.x = 1;
  grid_corner.y = 1;
  threads_corner.x = kr_nr;
  threads_corner.y = kr_nc;

  grid_lr_edge.x = grid.x;
  grid_lr_edge.y = 1;

  grid_tb_edge.x = 1;
  grid_tb_edge.y = grid.y;

  threads_lr_edge.x = threads.x;
  threads_lr_edge.y = kr_nc;

  threads_tb_edge.x = kr_nr;
  threads_tb_edge.y = threads.y;

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               center, full_supprt);

  k_corr2dnd<<<grid_corner, threads_corner, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               corners, full_supprt);

  k_corr2dnd<<<grid_lr_edge, threads_lr_edge, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               left_right_edge, full_supprt);

  k_corr2dnd<<<grid_tb_edge, threads_tb_edge, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               top_bottom_edge, full_supprt);

}

void
corr_2d_no_decimation_same_nrow_even_ncol_even(
  float const*        in,
  float const*        kr,
  float*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads, grid_corner, threads_corner, grid_lr_edge,
       threads_lr_edge, grid_tb_edge, threads_tb_edge;

  int dtr = -rshift - 1;
  int dtc = -cshift - 1;

  size_t const shared_memory_launch_size = 16000;

  distribute_matrix(in_nc - kr_nc + 1, in_nr - kr_nr + 1, grid, threads);
  grid_corner.x = 1;
  grid_corner.y = 1;
  threads_corner.x = kr_nr;
  threads_corner.y = kr_nc;

  grid_lr_edge.x = (in_nr - kr_nr + int(Dev_props::max_threads_per_block_x())) /
                                    int(Dev_props::max_threads_per_block_x());
  grid_lr_edge.y = 1;

  grid_tb_edge.x = 1;
  grid_tb_edge.y = (in_nc - kr_nc + int(Dev_props::max_threads_per_block_y())) /
                                    int(Dev_props::max_threads_per_block_y());

  threads_lr_edge.x = int(Dev_props::max_threads_per_block_x());
  threads_lr_edge.y = kr_nc;

  threads_tb_edge.x = kr_nr;
  threads_tb_edge.y = int(Dev_props::max_threads_per_block_y());

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               center, same_supprt_nrows_even_ncols_even);

  k_corr2dnd<<<grid_corner, threads_corner, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               corners, same_supprt_nrows_even_ncols_even);

  k_corr2dnd<<<grid_lr_edge, threads_lr_edge, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               left_right_edge, same_supprt_nrows_even_ncols_even);

  k_corr2dnd<<<grid_tb_edge, threads_tb_edge, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc),  int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               top_bottom_edge, same_supprt_nrows_even_ncols_even);

}

void
corr_2d_no_decimation_same_nrow_even_ncol_even(
  std::complex<float> const*        in,
  std::complex<float> const*        kr,
  std::complex<float>*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads, grid_corner, threads_corner, grid_lr_edge,
       threads_lr_edge, grid_tb_edge, threads_tb_edge;

  int dtr = -rshift - 1;
  int dtc = -cshift - 1;

  size_t const shared_memory_launch_size = 16000;

  threads.x = min(int(Dev_props::max_threads_per_block_x()),
                  int(sqrt(float(Dev_props::shared_memory_size() - 72) /
                                (8.0 * 2.5))));
  threads.y = threads.x / 2;

  grid.x = (in_nr - kr_nr + threads.x) / threads.x;
  grid.y = (in_nc - kr_nc + threads.y) / threads.y;

  grid_corner.x = 1;
  grid_corner.y = 1;
  threads_corner.x = kr_nr;
  threads_corner.y = kr_nc;

  grid_lr_edge.x = grid.x;
  grid_lr_edge.y = 1;

  grid_tb_edge.x = 1;
  grid_tb_edge.y = grid.y;

  threads_lr_edge.x = threads.x;
  threads_lr_edge.y = kr_nc;

  threads_tb_edge.x = kr_nr;
  threads_tb_edge.y = threads.y;

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               center, same_supprt_nrows_even_ncols_even);

  k_corr2dnd<<<grid_corner, threads_corner, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               corners, same_supprt_nrows_even_ncols_even);

  k_corr2dnd<<<grid_lr_edge, threads_lr_edge, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               left_right_edge, same_supprt_nrows_even_ncols_even);

  k_corr2dnd<<<grid_tb_edge, threads_tb_edge, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               top_bottom_edge, same_supprt_nrows_even_ncols_even);

}

void
corr_2d_no_decimation_same_nrow_even_ncol_odd(
  float const*        in,
  float const*        kr,
  float*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads, grid_corner, threads_corner, grid_lr_edge,
       threads_lr_edge, grid_tb_edge, threads_tb_edge;

  int dtr = -rshift - 1;
  int dtc = -cshift;

  size_t const shared_memory_launch_size = 16000;

  distribute_matrix(in_nc - kr_nc + 1, in_nr - kr_nr + 1, grid, threads);
  grid_corner.x = 1;
  grid_corner.y = 1;
  threads_corner.x = kr_nr;
  threads_corner.y = kr_nc;

  grid_lr_edge.x = (in_nr - kr_nr + int(Dev_props::max_threads_per_block_x())) /
                                    int(Dev_props::max_threads_per_block_x());
  grid_lr_edge.y = 1;

  grid_tb_edge.x = 1;
  grid_tb_edge.y = (in_nc - kr_nc + int(Dev_props::max_threads_per_block_y())) /
                                    int(Dev_props::max_threads_per_block_y());

  threads_lr_edge.x = int(Dev_props::max_threads_per_block_x());
  threads_lr_edge.y = kr_nc;

  threads_tb_edge.x = kr_nr;
  threads_tb_edge.y = int(Dev_props::max_threads_per_block_y());

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               center, same_supprt_nrows_even_ncols_odd);

  k_corr2dnd<<<grid_corner, threads_corner, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               corners, same_supprt_nrows_even_ncols_odd);

  k_corr2dnd<<<grid_lr_edge, threads_lr_edge, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               left_right_edge, same_supprt_nrows_even_ncols_odd);

  k_corr2dnd<<<grid_tb_edge, threads_tb_edge, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               top_bottom_edge, same_supprt_nrows_even_ncols_odd);

}

void
corr_2d_no_decimation_same_nrow_even_ncol_odd(
  std::complex<float> const*        in,
  std::complex<float> const*        kr,
  std::complex<float>*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads, grid_corner, threads_corner, grid_lr_edge,
       threads_lr_edge, grid_tb_edge, threads_tb_edge;

  int dtr = -rshift - 1;
  int dtc = -cshift;

  size_t const shared_memory_launch_size = 16000;

  threads.x = min(int(Dev_props::max_threads_per_block_x()),
                  int(sqrt(float(Dev_props::shared_memory_size() - 72) /
                                (8.0 * 2.5))));
  threads.y = threads.x / 2;

  grid.x = (in_nr - kr_nr + threads.x) / threads.x;
  grid.y = (in_nc - kr_nc + threads.y) / threads.y;

  grid_corner.x = 1;
  grid_corner.y = 1;
  threads_corner.x = kr_nr;
  threads_corner.y = kr_nc;

  grid_lr_edge.x = grid.x;
  grid_lr_edge.y = 1;

  grid_tb_edge.x = 1;
  grid_tb_edge.y = grid.y;

  threads_lr_edge.x = threads.x;
  threads_lr_edge.y = kr_nc;

  threads_tb_edge.x = kr_nr;
  threads_tb_edge.y = threads.y;

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               center, same_supprt_nrows_even_ncols_odd);

  k_corr2dnd<<<grid_corner, threads_corner, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               corners, same_supprt_nrows_even_ncols_odd);

  k_corr2dnd<<<grid_lr_edge, threads_lr_edge, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               left_right_edge, same_supprt_nrows_even_ncols_odd);

  k_corr2dnd<<<grid_tb_edge, threads_tb_edge, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               top_bottom_edge, same_supprt_nrows_even_ncols_odd);

}
void
corr_2d_no_decimation_same_nrow_odd_ncol_even(
  float const*        in,
  float const*        kr,
  float*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads, grid_corner, threads_corner, grid_lr_edge,
       threads_lr_edge, grid_tb_edge, threads_tb_edge;

  int dtr = -rshift;
  int dtc = -cshift - 1;

  size_t const shared_memory_launch_size = 16000;

  distribute_matrix(in_nc - kr_nc + 1, in_nr - kr_nr + 1, grid, threads);
  grid_corner.x = 1;
  grid_corner.y = 1;
  threads_corner.x = kr_nr;
  threads_corner.y = kr_nc;

  grid_lr_edge.x = (in_nr - kr_nr + int(Dev_props::max_threads_per_block_x())) /
                                    int(Dev_props::max_threads_per_block_x());
  grid_lr_edge.y = 1;

  grid_tb_edge.x = 1;
  grid_tb_edge.y = (in_nc - kr_nc + int(Dev_props::max_threads_per_block_y())) /
                                    int(Dev_props::max_threads_per_block_y());

  threads_lr_edge.x = int(Dev_props::max_threads_per_block_x());
  threads_lr_edge.y = kr_nc;

  threads_tb_edge.x = kr_nr;
  threads_tb_edge.y = int(Dev_props::max_threads_per_block_y());

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               center, same_supprt_nrows_odd_ncols_even);

  k_corr2dnd<<<grid_corner, threads_corner, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               corners, same_supprt_nrows_odd_ncols_even);

  k_corr2dnd<<<grid_lr_edge, threads_lr_edge, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               left_right_edge, same_supprt_nrows_odd_ncols_even);

  k_corr2dnd<<<grid_tb_edge, threads_tb_edge, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               top_bottom_edge, same_supprt_nrows_odd_ncols_even);

}

void
corr_2d_no_decimation_same_nrow_odd_ncol_even(
  std::complex<float> const*        in,
  std::complex<float> const*        kr,
  std::complex<float>*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads, grid_corner, threads_corner, grid_lr_edge,
       threads_lr_edge, grid_tb_edge, threads_tb_edge;

  int dtr = -rshift;
  int dtc = -cshift - 1;

  size_t const shared_memory_launch_size = 16000;

  threads.x = min(int(Dev_props::max_threads_per_block_x()),
                  int(sqrt(float(Dev_props::shared_memory_size() - 72) /
                                (8.0 * 2.5))));
  threads.y = threads.x / 2;

  grid.x = (in_nr - kr_nr + threads.x) / threads.x;
  grid.y = (in_nc - kr_nc + threads.y) / threads.y;


  grid_corner.x = 1;
  grid_corner.y = 1;
  threads_corner.x = kr_nr;
  threads_corner.y = kr_nc;

  grid_lr_edge.x = grid.x;
  grid_lr_edge.y = 1;

  grid_tb_edge.x = 1;
  grid_tb_edge.y = grid.y;

  threads_lr_edge.x = threads.x;
  threads_lr_edge.y = kr_nc;

  threads_tb_edge.x = kr_nr;
  threads_tb_edge.y = threads.y;

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               center, same_supprt_nrows_odd_ncols_even);

  k_corr2dnd<<<grid_corner, threads_corner, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               corners, same_supprt_nrows_odd_ncols_even);

  k_corr2dnd<<<grid_lr_edge, threads_lr_edge, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               left_right_edge, same_supprt_nrows_odd_ncols_even);

  k_corr2dnd<<<grid_tb_edge, threads_tb_edge, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               top_bottom_edge, same_supprt_nrows_odd_ncols_even);
}

void
corr_2d_no_decimation_same_nrow_odd_ncol_odd(
  float const*        in,
  float const*        kr,
  float*              out,
  size_t              in_nr,
  size_t              in_nc,
  size_t              kr_nr,
  size_t              kr_nc,
  size_t              ou_nr,
  size_t              ou_nc,
  size_t              rshift,
  size_t              cshift,
  int                 bias)
{
  dim3 grid, threads, grid_corner, threads_corner, grid_lr_edge,
       threads_lr_edge, grid_tb_edge, threads_tb_edge;

  int dtr = -rshift;
  int dtc = -cshift;

  size_t const shared_memory_launch_size = 16000;

  distribute_matrix(in_nc - kr_nc + 1, in_nr - kr_nr + 1, grid, threads);
  
  grid_corner.x = 1;
  grid_corner.y = 1;
  threads_corner.x = kr_nr;
  threads_corner.y = kr_nc;

  grid_lr_edge.x = (in_nr - kr_nr + int(Dev_props::max_threads_per_block_x())) /
                                    int(Dev_props::max_threads_per_block_x());
  grid_lr_edge.y = 1;

  grid_tb_edge.x = 1;
  grid_tb_edge.y = (in_nc - kr_nc + int(Dev_props::max_threads_per_block_y())) /
                                    int(Dev_props::max_threads_per_block_y());

  threads_lr_edge.x = int(Dev_props::max_threads_per_block_x());
  threads_lr_edge.y = kr_nc;

  threads_tb_edge.x = kr_nr;
  threads_tb_edge.y = int(Dev_props::max_threads_per_block_y());

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), int(dtr),
               int(dtc), bias, center, same_supprt_nrows_odd_ncols_odd);

  k_corr2dnd<<<grid_corner, threads_corner, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), int(dtr),
               int(dtc), bias, corners, same_supprt_nrows_odd_ncols_odd);

  k_corr2dnd<<<grid_lr_edge, threads_lr_edge, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), int(dtr),
               int(dtc), bias, left_right_edge, same_supprt_nrows_odd_ncols_odd);

  k_corr2dnd<<<grid_tb_edge, threads_tb_edge, shared_memory_launch_size>>>(
               in, kr, out, int(in_nr), int(in_nc), int(kr_nr), int(kr_nc),
               int(ou_nr), int(ou_nc), int(rshift), int(cshift), int(dtr),
               int(dtc), bias, top_bottom_edge, same_supprt_nrows_odd_ncols_odd);

}

void
corr_2d_no_decimation_same_nrow_odd_ncol_odd(
  std::complex<float> const*        in,
  std::complex<float> const*        kr,
  std::complex<float>*              out,
  size_t                            in_nr,
  size_t                            in_nc,
  size_t                            kr_nr,
  size_t                            kr_nc,
  size_t                            ou_nr,
  size_t                            ou_nc,
  size_t                            rshift,
  size_t                            cshift,
  int                 bias)
{
  dim3 grid, threads, grid_corner, threads_corner, grid_lr_edge,
       threads_lr_edge, grid_tb_edge, threads_tb_edge;

  int dtr = -rshift;
  int dtc = -cshift;

  size_t const shared_memory_launch_size = 16000;

  threads.x = min(int(Dev_props::max_threads_per_block_x()),
                  int(sqrt(float(Dev_props::shared_memory_size() - 72) /
                                (8.0 * 2.5))));
  threads.y = threads.x / 2;

  grid.x = (in_nr - kr_nr + threads.x) / threads.x;
  grid.y = (in_nc - kr_nc + threads.y) / threads.y;


  grid_corner.x = 1;
  grid_corner.y = 1;
  threads_corner.x = kr_nr;
  threads_corner.y = kr_nc;

  grid_lr_edge.x = grid.x;
  grid_lr_edge.y = 1;

  grid_tb_edge.x = 1;
  grid_tb_edge.y = grid.y;

  threads_lr_edge.x = threads.x;
  threads_lr_edge.y = kr_nc;

  threads_tb_edge.x = kr_nr;
  threads_tb_edge.y = threads.y;

  k_corr2dnd<<<grid, threads, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               center, same_supprt_nrows_odd_ncols_odd);

  k_corr2dnd<<<grid_corner, threads_corner, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               corners, same_supprt_nrows_odd_ncols_odd);

  k_corr2dnd<<<grid_lr_edge, threads_lr_edge, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               left_right_edge, same_supprt_nrows_odd_ncols_odd);

  k_corr2dnd<<<grid_tb_edge, threads_tb_edge, shared_memory_launch_size>>>(
               reinterpret_cast<cuComplex const*>(in),
               reinterpret_cast<cuComplex const*>(kr),
               reinterpret_cast<cuComplex*>(out), int(in_nr),
               int(in_nc), int(kr_nr), int(kr_nc), int(ou_nr),
               int(ou_nc), int(rshift), int(cshift), dtr, dtc, bias,
               top_bottom_edge, same_supprt_nrows_odd_ncols_odd);

}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

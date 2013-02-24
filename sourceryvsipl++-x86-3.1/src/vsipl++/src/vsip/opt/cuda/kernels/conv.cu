/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA kernels for convolution and correlation functions.
#include <cuComplex.h>
#include <complex>
#include "util.hpp"
#include "cmplx.cuh"

using namespace dev;

// 1-D convolution/correlation for real input/real coefficients -> real output.
__global__ void
k_conv_ss(
  float const* input,
  float const* kernel,
  float*       out,
  size_t       input_len,
  size_t       kernel_len,
  size_t       output_len,
  size_t       decm,
  size_t       shift,
  bool         is_conv,
  bool         is_even,
  bool         is_same,
  bool         is_min,
  int          bias)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;
  int tid_decm = __mul24(tid, decm);

  float sum = 0;

  for (unsigned int i = 0; i < kernel_len; ++i)
  {
    int index = (is_conv ? tid_decm - i + shift :
                 tid_decm + i + shift - (kernel_len - 1 +
                                        (is_even && is_same ? 1 : 0)));
    if (index >= 0 && index < input_len)
      sum += input[index] * kernel[i];
  }

  if (bias)
  {
    if (tid_decm < kernel_len - 1 - shift + (is_even && (!is_min) ? 1 : 0))
    {
      sum /= float(tid_decm + 1 + shift -
                  (is_even && (is_same || is_min) ? 1 : 0));
    }
    else if (tid_decm >= input_len - shift)
      sum /= float(input_len - tid_decm - 1 + kernel_len - shift);
    else
      sum /= float(kernel_len);
  } 
    
  if (tid < output_len)
    *(out + tid) = sum;
}

// 2-D convolution/correlation for real input/real coefficients -> real output.
__global__ void
k_conv_2d_ss(
  float const* input,
  float const* kernel,
  float*       out,
  size_t       input_nrows,
  size_t       input_ncols,
  size_t       kernel_nrows,
  size_t       kernel_ncols,
  size_t       output_nrows,
  size_t       output_ncols,
  size_t       row_decm,
  size_t       col_decm,
  size_t       row_stride,
  ptrdiff_t    row_shift,
  ptrdiff_t    col_shift,
  bool         is_conv,
  int          bias)
{
  int tr = threadIdx.x;
  int tc = threadIdx.y;
  int br = blockIdx.x;
  int bc = blockIdx.y;
  int tidr = __mul24(blockDim.x, br) + tr;
  int tidc = __mul24(blockDim.y, bc) + tc;
  int tidr_decm = __mul24(tidr, col_decm);
  int tidc_decm = __mul24(tidc, row_decm);

  float sum_row = 0;
  float sum = 0;
  float scale = 1;

  for (unsigned int i = 0; i < kernel_nrows; ++i)
  {
    sum_row = 0;
    for (unsigned int j = 0; j < kernel_ncols; ++j)
    {
      int row_index = (is_conv ? tidr_decm - i + row_shift :
                                 tidr_decm + i + row_shift);
      int col_index = (is_conv ? tidc_decm - j + col_shift :
                                 tidc_decm + j + col_shift);
      
      if (row_index >= 0 && row_index < input_nrows &&
          col_index >= 0 && col_index < input_ncols)
        sum_row += (*(input + row_index * row_stride + col_index)) *
                   (*(kernel + i * kernel_ncols + j));
    }
    sum += sum_row;
  }

  if (bias)
  {
    if (tidr_decm + row_shift < 0) 
      scale *= float(tidr_decm + row_shift + kernel_nrows);
    else if (tidr_decm + row_shift + kernel_nrows > input_nrows)
      scale *= float(input_nrows - tidr_decm - row_shift);
    else
      scale *= float(kernel_nrows);

    if (tidc_decm + col_shift < 0)
      scale *= float(kernel_ncols + tidc_decm + col_shift);
    else if (tidc_decm + col_shift + kernel_ncols > input_ncols)
      scale *= float(input_ncols - tidc_decm - col_shift);
    else
      scale *= float(kernel_ncols);

    sum /= scale;
  }

  if (tidc < output_ncols && tidr < output_nrows)
    *(out + tidr * output_ncols + tidc) = sum;
}

// 1-D convolution/correlation for complex input/complex
//   coefficients -> complex output.
__global__ void
k_conv_cc(
  cuComplex const* input,
  cuComplex const* kernel,
  cuComplex*       out,
  size_t           input_len,
  size_t           kernel_len,
  size_t           output_len,
  size_t           decm,
  size_t           shift,
  bool             is_conv,
  bool             is_even,
  bool             is_same,
  bool             is_min,
  int              bias)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int tid = __mul24(blockDim.x, bx) + tx;
  int tid_decm = __mul24(tid, decm);
  float conj_fact = (is_conv ? 1.0 : -1.0);

  cuComplex sum;
  cuComplex temp;

  sum.x = 0;
  sum.y = 0;

  for (unsigned int i = 0; i < kernel_len; ++i)
  {
    int index = (is_conv ? tid_decm - i + shift :
                           tid_decm + i + shift -
                            (kernel_len - 1 + (is_even && is_same ? 1 : 0)));
    if (index >= 0 && index < input_len)
    {
      cmulco(temp, kernel[i],  input[index], conj_fact);
      sum.x += temp.x;
      sum.y += temp.y;
    }
  }

  if (bias)
  {
    if (tid_decm < kernel_len - 1 - shift + (is_even && (!is_min) ? 1 : 0))
    {
      sum.x /= float(tid_decm + 1 + shift -
                    (is_even && (is_same || is_min) ? 1 : 0));
      sum.y /= float(tid_decm + 1 + shift -
                    (is_even && (is_same || is_min) ? 1 : 0));
    }
    else if (tid_decm >= input_len - shift)
    {
      sum.x /= float(input_len - tid_decm - 1 + kernel_len - shift);
      sum.y /= float(input_len - tid_decm - 1 + kernel_len - shift);
    }
    else
    {
      sum.x /= float(kernel_len);
      sum.y /= float(kernel_len);
    }
  } 

  if (tid < output_len)
    *(out + tid) = sum;
}

// 2-D convolution/correlation for complex input/complex
//  coefficients -> complex output
__global__ void
k_conv_2d_cc(
  cuComplex const* input,
  cuComplex const* kernel,
  cuComplex*       out,
  size_t           input_nrows,
  size_t           input_ncols,
  size_t           kernel_nrows,
  size_t           kernel_ncols,
  size_t           output_nrows,
  size_t           output_ncols,
  size_t           row_decm,
  size_t           col_decm,
  size_t           row_stride,
  ptrdiff_t        row_shift,
  ptrdiff_t        col_shift,
  bool             is_conv,
  int              bias)
{
  int tr = threadIdx.x;
  int tc = threadIdx.y;
  int br = blockIdx.x;
  int bc = blockIdx.y;
  int tidr = __mul24(blockDim.x, br) + tr;
  int tidc = __mul24(blockDim.y, bc) + tc;
  int tidr_decm = __mul24(tidr, col_decm);
  int tidc_decm = __mul24(tidc, row_decm);
  float conj_fact = (is_conv ? 1.0: -1.0);
  float scale = 1;

  cuComplex sum_row;
  cuComplex sum;
  cuComplex temp;

  sum.x = 0;
  sum.y = 0;

  for (unsigned int i = 0; i < kernel_nrows; ++i)
  {
    sum_row.x = 0;
    sum_row.y = 0;
    for (unsigned int j = 0; j < kernel_ncols; ++j)
    {
      int row_index = (is_conv ? tidr_decm - i + row_shift :
                                 tidr_decm + i + row_shift);
      int col_index = (is_conv ? tidc_decm - j + col_shift :
                                 tidc_decm + j + col_shift);
      if (row_index >= 0 && row_index < input_nrows &&
          col_index >= 0 && col_index < input_ncols)
      {
        cmulco(temp, *(kernel + i * kernel_ncols + j),
                    *(input + row_index * row_stride + col_index),  conj_fact);
        sum_row.x += temp.x;
        sum_row.y += temp.y;
      }
    }
    sum.x += sum_row.x;
    sum.y += sum_row.y;
  }

  if (bias)
  {

    if (tidr_decm + row_shift < 0) 
      scale *= float(tidr_decm + row_shift + kernel_nrows);
    else if (tidr_decm + row_shift + kernel_nrows > input_nrows)
      scale *= float(input_nrows - tidr_decm - row_shift);
    else
      scale *= float(kernel_nrows);

    if (tidc_decm + col_shift < 0)
      scale *= float(kernel_ncols + tidc_decm + col_shift);
    else if (tidc_decm + col_shift + kernel_ncols > input_ncols)
      scale *= float(input_ncols - tidc_decm - col_shift);
    else
      scale *= float(kernel_ncols);
    
    sum.x /= scale;
    sum.y /= scale;
  }

  if (tidc < output_ncols && tidr < output_nrows)
    *(out + tidr * output_ncols + tidc) = sum;
}

namespace vsip
{
namespace impl
{
namespace cuda
{

void
conv(
  float const*     in,
  float const*     kr,
  float*           out,
  size_t           in_len,
  size_t           kr_len,
  size_t           ou_len,
  size_t           dec,
  size_t           shift,
  bool             iscv,
  bool             isev,
  bool             issm,
  bool             ismn,
  int              bias)
{
  dim3 grid, threads;
  distribute_vector(ou_len, grid, threads);

  k_conv_ss<<<grid, threads>>>(in, kr, out, in_len, kr_len, ou_len, dec, shift,
                               iscv, isev, issm, ismn, bias);

}

void
conv(
  std::complex<float> const* in,
  std::complex<float> const* kr,
  std::complex<float>*       out,
  size_t                     in_len,
  size_t                     kr_len,
  size_t                     ou_len,
  size_t                     dec,
  size_t                     shift,
  bool                       iscv,
  bool                       isev,
  bool                       issm,
  bool                       ismn,
  int                        bias)
{
  dim3 grid, threads;
  distribute_vector(ou_len, grid, threads);

  k_conv_cc<<<grid, threads>>>(reinterpret_cast<cuComplex const*>(in),
                               reinterpret_cast<cuComplex const*>(kr),
                               reinterpret_cast<cuComplex*>(out), in_len,
                               kr_len, ou_len, dec, shift, iscv, isev, issm,
                               ismn, bias);

}

void
conv_2d(
  float const*     in,
  float const*     kr,
  float*           out,
  size_t           in_nr,
  size_t           in_nc,
  size_t           kr_nr,
  size_t           kr_nc,
  size_t           ou_nr,
  size_t           ou_nc,
  size_t           rdec,
  size_t           cdec,
  size_t           rstride,
  ptrdiff_t        rshift,
  ptrdiff_t        cshift,
  bool             iscv,
  int              bias)
{
  dim3 grid, threads;
  distribute_matrix(ou_nc, ou_nr, grid, threads);

  k_conv_2d_ss<<<grid, threads>>>(in, kr, out, in_nr, in_nc, kr_nr,
                                  kr_nc, ou_nr, ou_nc, rdec, cdec,
                                  rstride, rshift, cshift, iscv, bias);

}

void
conv_2d(
  std::complex<float> const*     in,
  std::complex<float> const*     kr,
  std::complex<float>*           out,
  size_t                         in_nr,
  size_t                         in_nc,
  size_t                         kr_nr,
  size_t                         kr_nc,
  size_t                         ou_nr,
  size_t                         ou_nc,
  size_t                         rdec,
  size_t                         cdec,
  size_t                         rstride,
  ptrdiff_t                      rshift,
  ptrdiff_t                      cshift,
  bool                           iscv,
  int                            bias)
{
  dim3 grid, threads;
  distribute_matrix(ou_nc, ou_nr, grid, threads);

  k_conv_2d_cc<<<grid, threads>>>(reinterpret_cast<cuComplex const*>(in),
                                  reinterpret_cast<cuComplex const*>(kr),
                                  reinterpret_cast<cuComplex*>(out), in_nr,
                                  in_nc, kr_nr, kr_nc, ou_nr, ou_nc, rdec,
                                  cdec, rstride, rshift, cshift, iscv, bias);

}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

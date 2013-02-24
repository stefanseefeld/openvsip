/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for Fast Fourier Transforms using CUDA

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip/opt/cuda/kernels.hpp>
#include <vsip_csl/error_db.hpp>

#include "benchmarks.hpp"
#include "fastconv.hpp"


using namespace vsip;


float
fft_ops(length_type len)
{
  return 5.0 * len * std::log((double)len) / std::log(2.0);
}


template <typename T,
	  typename Tag>
struct t_fftm;

struct Impl_op;		// out-of-place
struct Impl_ip;		// in-place
struct Impl_dev;	// On-device (memory moves are not timed)



/***********************************************************************
  Impl_op: Out-of-place Fftm
***********************************************************************/

template <>
struct t_fftm<std::complex<float>, Impl_op> : Benchmark_base
{
  typedef std::complex<float>  T;
  static int const elem_per_point = 2;

  typedef Map<Block_dist, Whole_dist>      map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;
  typedef Matrix<T, block_type>            view_type;

  char const* what() { return "CUDA t_fftm<complex<float>, Impl_op>"; }
  float ops(length_type rows, length_type cols)
    { return rows * fft_ops(cols); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    view_type data_in(rows, cols, T());
    view_type data_out(rows, cols, T());

    {
      vsip::dda::Data<block_type, vsip::dda::in> ext_in(data_in.block());
      vsip::dda::Data<block_type, vsip::dda::out> ext_out(data_out.block());
      T const *p_in = ext_in.ptr();
      T *p_out = ext_out.ptr();

      cudaError_t error;
      cufftComplex* d_in = NULL;
      error = cudaMalloc((void**)&d_in, rows*cols*sizeof(T));
      test_assert(error == cudaSuccess);
      test_assert(d_in != NULL);

      cufftComplex* d_out = NULL;
      error = cudaMalloc((void**)&d_out, rows*cols*sizeof(T));
      test_assert(error == cudaSuccess);
      test_assert(d_out != NULL);


      // There is an 8-million element limit to 1-D FFT(M)s.
      // Determine how many sub-rows can be done at a time under this limit.
      unsigned int sr = rows;
      while (sr > 8000000 / cols)
        sr /= 2;
      test_assert(sr > 0);

      cufftHandle plan;
      cufftResult result;
      result = cufftPlan1d(&plan, cols, CUFFT_C2C, sr);
      test_assert(result == CUFFT_SUCCESS);

      
      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l=0; l<loop; ++l)
      {
        // Copy data to device memory
        error = cudaMemcpy(d_in, p_in, rows*cols*sizeof(T), cudaMemcpyHostToDevice);
        test_assert(error == cudaSuccess);

        // Perform FFT
        for (length_type r = 0; r < rows; r += sr)
        {
          index_type row = r * sr * cols;
          result = cufftExecC2C(plan, &d_in[row], &d_out[row], CUFFT_FORWARD);
          test_assert(result == CUFFT_SUCCESS);
        }
        cudaThreadSynchronize();

        // Copy data from device memory
        error = cudaMemcpy(p_out, d_out, rows*cols*sizeof(T), cudaMemcpyDeviceToHost);
        test_assert(error == cudaSuccess);
      }
      t1.stop();
      time = t1.delta();


      cufftDestroy(plan);
      cudaFree(d_out);
      cudaFree(d_in);
    }
  }

  // Member data
};


/***********************************************************************
  Impl_ip: In-place Fftm
***********************************************************************/

template <>
struct t_fftm<std::complex<float>, Impl_ip> : Benchmark_base
{
  typedef std::complex<float>  T;
  static int const elem_per_point = 2;

  typedef Map<Block_dist, Whole_dist>      map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;
  typedef Matrix<T, block_type>            view_type;

  char const* what() { return "CUDA t_fftm<complex<float>, Impl_op>"; }
  float ops(length_type rows, length_type cols)
    { return rows * fft_ops(cols); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    view_type data(rows, cols, T());

    {
      vsip::dda::Data<block_type, vsip::dda::inout> ext_data(data.block());
      T *p_data = ext_data.ptr();

      cudaError_t error;
      cufftComplex* d_data = NULL;
      error = cudaMalloc((void**)&d_data, rows*cols*sizeof(T));
      test_assert(error == cudaSuccess);
      test_assert(d_data != NULL);


      // There is an 8-million element limit to 1-D FFT(M)s.
      // Determine how many sub-rows can be done at a time under this limit.
      unsigned int sr = rows;
      while (sr > 8000000 / cols)
        sr /= 2;
      test_assert(sr > 0);

      cufftHandle plan;
      cufftResult result;
      result = cufftPlan1d(&plan, cols, CUFFT_C2C, sr);
      test_assert(result == CUFFT_SUCCESS);

      
      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l=0; l<loop; ++l)
      {
        // Copy data to device memory
        error = cudaMemcpy(d_data, p_data, rows*cols*sizeof(T), cudaMemcpyHostToDevice);
        test_assert(error == cudaSuccess);

        // Perform FFT
        for (length_type r = 0; r < rows; r += sr)
        {
          index_type row = r * sr * cols;
          result = cufftExecC2C(plan, &d_data[row], &d_data[row], CUFFT_FORWARD);
          test_assert(result == CUFFT_SUCCESS);
        }
        cudaThreadSynchronize();

        // Copy data from device memory
        error = cudaMemcpy(p_data, d_data, rows*cols*sizeof(T), cudaMemcpyDeviceToHost);
        test_assert(error == cudaSuccess);
      }
      t1.stop();
      time = t1.delta();


      cufftDestroy(plan);
      cudaFree(d_data);
    }
  }

  // Member data
};


/***********************************************************************
  Impl_dev: On-device FFTM (no memory moves)
***********************************************************************/

template <>
struct t_fftm<std::complex<float>, Impl_dev> : Benchmark_base
{
  typedef std::complex<float>  T;
  static int const elem_per_point = 2;

  typedef Map<Block_dist, Whole_dist>      map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;
  typedef Matrix<T, block_type>            view_type;

  char const* what() { return "CUDA t_fftm<complex<float>, Impl_op>"; }
  float ops(length_type rows, length_type cols)
    { return rows * fft_ops(cols); }

  void fftm(length_type rows, length_type cols, length_type loop, float& time)
  {
    view_type data(rows, cols, T());

    {
      vsip::dda::Data<block_type, vsip::dda::inout> ext_data(data.block());
      T *p_data = ext_data.ptr();

      cudaError_t error;
      cufftComplex* d_data = NULL;
      error = cudaMalloc((void**)&d_data, rows*cols*sizeof(T));
      test_assert(error == cudaSuccess);
      test_assert(d_data != NULL);


      // There is an 8-million element limit to 1-D FFT(M)s.
      // Determine how many sub-rows can be done at a time under this limit.
      unsigned int sr = rows;
      while (sr > 8000000 / cols)
        sr /= 2;
      test_assert(sr > 0);

      cufftHandle plan;
      cufftResult result;
      result = cufftPlan1d(&plan, cols, CUFFT_C2C, sr);
      test_assert(result == CUFFT_SUCCESS);

      // Copy data to device memory
      error = cudaMemcpy(d_data, p_data, rows*cols*sizeof(T), cudaMemcpyHostToDevice);
      test_assert(error == cudaSuccess);

      
      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l=0; l<loop; ++l)
      {
        // Perform FFT
        for (length_type r = 0; r < rows; r += sr)
        {
          index_type row = r * sr * cols;
          result = cufftExecC2C(plan, &d_data[row], &d_data[row], CUFFT_FORWARD);
          test_assert(result == CUFFT_SUCCESS);
        }
        cudaThreadSynchronize();
      }
      t1.stop();
      time = t1.delta();


      // Copy data from device memory
      error = cudaMemcpy(p_data, d_data, rows*cols*sizeof(T), cudaMemcpyDeviceToHost);
      test_assert(error == cudaSuccess);

      cufftDestroy(plan);
      cudaFree(d_data);
    }
  }

  // Member data
};



/***********************************************************************
  Fixed rows driver
***********************************************************************/

template <typename T, typename ImplTag>
struct t_fftm_fix_rows : public t_fftm<T, ImplTag>
{
  typedef t_fftm<T, ImplTag> base_type;
  static int const elem_per_point = base_type::elem_per_point;

  char const* what() { return "CUDA t_fftm_fix_rows"; }
  float ops_per_point(length_type cols)
    { return (int)(this->ops(rows_, cols) / cols); }

  int riob_per_point(length_type) { return rows_*sizeof(T); }
  int wiob_per_point(length_type) { return rows_*sizeof(T); }
  int mem_per_point (length_type) { return rows_*elem_per_point*sizeof(T); }

  void operator()(length_type cols, length_type loop, float& time)
  {
    this->fftm(rows_, cols, loop, time);
  }

  t_fftm_fix_rows(length_type rows)
    : rows_(rows)
  {}

// Member data
  length_type rows_;
};



/***********************************************************************
  Fixed cols driver
***********************************************************************/

template <typename T, typename ImplTag>
struct t_fftm_fix_cols : public t_fftm<T, ImplTag>
{
  typedef t_fftm<T, ImplTag> base_type;
  static int const elem_per_point = base_type::elem_per_point;

  char const* what() { return "CUDA t_fftm_fix_cols"; }
  float ops_per_point(length_type rows)
    { return (int)(this->ops(rows, cols_) / rows); }
  int riob_per_point(length_type) { return cols_*sizeof(T); }
  int wiob_per_point(length_type) { return cols_*sizeof(T); }
  int mem_per_point (length_type) { return cols_*elem_per_point*sizeof(T); }

  void operator()(length_type rows, length_type loop, float& time)
  {
    this->fftm(rows, cols_, loop, time);
  }

  t_fftm_fix_cols(length_type cols)
    : cols_(cols)
  {}

// Member data
  length_type cols_;
};


/***********************************************************************
  Benchmark Driver
***********************************************************************/

void
defaults(Loop1P& loop)
{
  loop.cal_        = 4;
  loop.start_      = 4;
  loop.stop_       = 16;
  loop.loop_start_ = 10;

  loop.param_["rows"] = "64";
  loop.param_["size"] = "2048";
}



int
test(Loop1P& loop, int what)
{
  length_type rows  = atoi(loop.param_["rows"].c_str());
  length_type size  = atoi(loop.param_["size"].c_str());

  std::cout << "rows: " << rows << "  size: " << size 
	    << std::endl;

  switch (what)
  {
  case  1:  loop(t_fftm_fix_rows<complex<float>, Impl_op>(rows)); break;
  case  2:  loop(t_fftm_fix_rows<complex<float>, Impl_ip>(rows)); break;
  case  3:  loop(t_fftm_fix_rows<complex<float>, Impl_dev>(rows)); break;

  case  11: loop(t_fftm_fix_cols<complex<float>, Impl_op>(size)); break;
  case  12: loop(t_fftm_fix_cols<complex<float>, Impl_ip>(size)); break;
  case  13: loop(t_fftm_fix_cols<complex<float>, Impl_dev>(size)); break;

  case 0:
    std::cout
      << "fftm -- FFT/FFTM benchmark using CUDA\n"
      << " Fixed rows, sweeping FFT size:\n"
      << "   -1 -- op  : out-of-place CC fwd fft\n"
      << "   -2 -- ip  : In-place CC fwd fft\n"
      << "   -3 -- dev : On-device CC fwd fft\n"
      << "\n"
      << " Parameters (for sweeping FFT size, cases 1 through 6)\n"
      << "  -p:rows ROWS -- set number of pulses (default 64)\n"
      << "\n"
      << " Fixed FFT size, sweeping number of FFTs:\n"
      << "  -11 -- op  : out-of-place CC fwd fft\n"
      << "  -12 -- ip  : In-place CC fwd fft\n"
      << "  -13 -- dev : On-device CC fwd fft\n"
      << "\n"
      << " Parameters (for sweeping number of FFTs, cases 11 through 16)\n"
      << "  -p:size SIZE -- size of pulse (default 2048)\n"
      ;

  default: return 0;
  }
  return 1;
}

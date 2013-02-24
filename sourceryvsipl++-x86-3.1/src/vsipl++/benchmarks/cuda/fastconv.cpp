/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for Fast Convolution using CUDA.

#include <iostream>
#include <cuda_runtime_api.h>
#include <cufft.h>

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

struct Impl_mem_phased;		// out-of-place, phased fast-convolution
struct Impl_dev_phased;	        // on-device (no memory moves), phased
struct Impl_dev_interleaved;	// on-device (no memory moves), interleaved


/***********************************************************************
  Impl_mem_phased: out-of-place, phased fast-convolution
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl_mem_phased> : fastconv_ops
{
  static length_type const num_args = 2;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef Dense<2, T, row2_type> mat_block_type;
    typedef Matrix<T, mat_block_type> view_type;
    typedef Dense<1, T, row1_type> vec_block_type;
    typedef Vector<T, vec_block_type> replica_view_type;

    view_type data(npulse, nrange, T());

    // Create the pulse replica
    replica_view_type replica(nrange, T());

    {
      vsip::dda::Data<mat_block_type, vsip::dda::inout> ext_data(data.block());
      vsip::dda::Data<vec_block_type, vsip::dda::inout> ext_replica(replica.block());
      T *p_data = ext_data.ptr();
      T *p_replica = ext_replica.ptr();

      cudaError_t error;
      cufftComplex* d_data = NULL;
      error = cudaMalloc((void**)&d_data, npulse*nrange*sizeof(T));
      test_assert(error == cudaSuccess);
      test_assert(d_data != NULL);

      cufftComplex* d_replica = NULL;
      error = cudaMalloc((void**)&d_replica, nrange*sizeof(T));
      test_assert(error == cudaSuccess);
      test_assert(d_replica != NULL);
      error = cudaMemcpy(d_replica, p_replica, nrange*sizeof(T), cudaMemcpyHostToDevice);
      test_assert(error == cudaSuccess);


      // There is an 8-million element limit to 1-D FFT(M)s.
      // Determine how many sub-rows can be done at a time under this limit.
      unsigned int sr = npulse;
      while (sr > 8000000 / nrange)
        sr /= 2;
      test_assert(sr > 0);

      cufftHandle plan;
      cufftResult result;
      result = cufftPlan1d(&plan, nrange, CUFFT_C2C, sr);
      test_assert(result == CUFFT_SUCCESS);

      
      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l=0; l<loop; ++l)
      {
        // Perform fast convolution:
        error = cudaMemcpy(d_data, p_data, npulse*nrange*sizeof(T), cudaMemcpyHostToDevice);
        test_assert(error == cudaSuccess);

        for (length_type r = 0; r < npulse; r += sr)
        {
          index_type row = r * sr * nrange;
          result = cufftExecC2C(plan, &d_data[row], &d_data[row], CUFFT_FORWARD);
          test_assert(result == CUFFT_SUCCESS);
        }

        vsip::impl::cuda::vmmuls_row(
          p_replica, p_data, p_data, 1 / (float)nrange, npulse, nrange);

        for (length_type r = 0; r < npulse; r += sr)
        {
          index_type row = r * sr * nrange;
          result = cufftExecC2C(plan, &d_data[row], &d_data[row], CUFFT_INVERSE);
          test_assert(result == CUFFT_SUCCESS);
        }

        error = cudaMemcpy(p_data, d_data, npulse*nrange*sizeof(T), cudaMemcpyDeviceToHost);
        test_assert(error == cudaSuccess);
      }
      t1.stop();
      time = t1.delta();


      cufftDestroy(plan);
      cudaFree(d_replica);
      cudaFree(d_data);
    }

  }
};


/***********************************************************************
  Impl_dev_phased: On-device (no memory moves), phased
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl_dev_phased> : fastconv_ops
{
  static length_type const num_args = 2;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef Dense<2, T, row2_type> mat_block_type;
    typedef Matrix<T, mat_block_type> view_type;
    typedef Dense<1, T, row1_type> vec_block_type;
    typedef Vector<T, vec_block_type> replica_view_type;

    view_type data(npulse, nrange, T());

    // Create the pulse replica
    replica_view_type replica(nrange, T());

    {
      vsip::dda::Data<mat_block_type, vsip::dda::inout> ext_data(data.block());
      vsip::dda::Data<vec_block_type, vsip::dda::inout> ext_replica(replica.block());
      T *p_data = ext_data.ptr();
      T *p_replica = ext_replica.ptr();

      cudaError_t error;
      cufftComplex* d_data = NULL;
      error = cudaMalloc((void**)&d_data, npulse*nrange*sizeof(T));
      test_assert(error == cudaSuccess);
      test_assert(d_data != NULL);
      error = cudaMemcpy(d_data, p_data, npulse*nrange*sizeof(T), cudaMemcpyHostToDevice);
      test_assert(error == cudaSuccess);

      cufftComplex* d_replica = NULL;
      error = cudaMalloc((void**)&d_replica, nrange*sizeof(T));
      test_assert(error == cudaSuccess);
      test_assert(d_replica != NULL);
      error = cudaMemcpy(d_replica, p_replica, nrange*sizeof(T), cudaMemcpyHostToDevice);
      test_assert(error == cudaSuccess);


      // There is an 8-million element limit to 1-D FFT(M)s.
      // Determine how many sub-rows can be done at a time under this limit.
      unsigned int sr = npulse;
      while (sr > 8000000 / nrange)
        sr /= 2;
      test_assert(sr > 0);

      cufftHandle plan;
      cufftResult result;
      result = cufftPlan1d(&plan, nrange, CUFFT_C2C, sr);
      test_assert(result == CUFFT_SUCCESS);

      
      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l=0; l<loop; ++l)
      {
        // Perform fast convolution:
        for (length_type r = 0; r < npulse; r += sr)
        {
          index_type row = r * sr * nrange;
          result = cufftExecC2C(plan, &d_data[row], &d_data[row], CUFFT_FORWARD);
          test_assert(result == CUFFT_SUCCESS);
        }

        vsip::impl::cuda::vmmuls_row(
          p_replica, p_data, p_data, 1 / (float)nrange, npulse, nrange);
        cudaThreadSynchronize();

        for (length_type r = 0; r < npulse; r += sr)
        {
          index_type row = r * sr * nrange;
          result = cufftExecC2C(plan, &d_data[row], &d_data[row], CUFFT_INVERSE);
          test_assert(result == CUFFT_SUCCESS);
        }
      }
      t1.stop();
      time = t1.delta();

      // bring data back
      error = cudaMemcpy(p_data, d_data, npulse*nrange*sizeof(T), cudaMemcpyDeviceToHost);
      test_assert(error == cudaSuccess);

      cufftDestroy(plan);
      cudaFree(d_replica);
      cudaFree(d_data);
    }

  }
};


/***********************************************************************
  Impl_dev_interleaved: On-device (no memory moves), interleaved
***********************************************************************/

template <typename T>
struct t_fastconv_base<T, Impl_dev_interleaved> : fastconv_ops
{
  static length_type const num_args = 2;

  void fastconv(length_type npulse, length_type nrange,
		length_type loop, float& time)
  {
    typedef Dense<2, T, row2_type> mat_block_type;
    typedef Matrix<T, mat_block_type> view_type;
    typedef Dense<1, T, row1_type> vec_block_type;
    typedef Vector<T, vec_block_type> replica_view_type;

    view_type data(npulse, nrange, T());

    // Create the pulse replica
    replica_view_type replica(nrange, T());

    {
      vsip::dda::Data<mat_block_type, vsip::dda::inout> ext_data(data.block());
      vsip::dda::Data<vec_block_type, vsip::dda::inout> ext_replica(replica.block());
      T *p_data = ext_data.ptr();
      T *p_replica = ext_replica.ptr();

      cudaError_t error;
      cufftComplex* d_data = NULL;
      error = cudaMalloc((void**)&d_data, npulse*nrange*sizeof(T));
      test_assert(error == cudaSuccess);
      test_assert(d_data != NULL);
      error = cudaMemcpy(d_data, p_data, npulse*nrange*sizeof(T), cudaMemcpyHostToDevice);
      test_assert(error == cudaSuccess);

      cufftComplex* d_replica = NULL;
      error = cudaMalloc((void**)&d_replica, nrange*sizeof(T));
      test_assert(error == cudaSuccess);
      test_assert(d_replica != NULL);
      error = cudaMemcpy(d_replica, p_replica, nrange*sizeof(T), cudaMemcpyHostToDevice);
      test_assert(error == cudaSuccess);


      // FFT's are performed one at a time
      cufftHandle plan;
      cufftResult result;
      result = cufftPlan1d(&plan, nrange, CUFFT_C2C, 1);
      test_assert(result == CUFFT_SUCCESS);

      
      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l=0; l<loop; ++l)
      {
        // Perform fast convolution:
        for (length_type r = 0; r < npulse; ++r)
        {
          index_type row = r * nrange;
          result = cufftExecC2C(plan, &d_data[row], &d_data[row], CUFFT_FORWARD);
          test_assert(result == CUFFT_SUCCESS);

          vsip::impl::cuda::vmmuls_row(
            p_replica, &p_data[row], &p_data[row], 1 / (float)nrange, 1, nrange);

          result = cufftExecC2C(plan, &d_data[row], &d_data[row], CUFFT_INVERSE);
          test_assert(result == CUFFT_SUCCESS);
        }
      }
      t1.stop();
      time = t1.delta();

      // bring data back
      error = cudaMemcpy(p_data, d_data, npulse*nrange*sizeof(T), cudaMemcpyDeviceToHost);
      test_assert(error == cudaSuccess);

      cufftDestroy(plan);
      cudaFree(d_replica);
      cudaFree(d_data);
    }

  }
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
  case  1:  loop(t_fastconv_pf<complex<float>, Impl_mem_phased>(rows)); break;
  case  2:  loop(t_fastconv_pf<complex<float>, Impl_dev_phased>(rows)); break;
  case  3:  loop(t_fastconv_pf<complex<float>, Impl_dev_interleaved>(rows)); break;

  case  11: loop(t_fastconv_rf<complex<float>, Impl_mem_phased>(size)); break;
  case  12: loop(t_fastconv_rf<complex<float>, Impl_dev_phased>(size)); break;
  case  13: loop(t_fastconv_rf<complex<float>, Impl_dev_interleaved>(rows)); break;

  case 0:
    std::cout
      << "fastconv -- fast convolution benchmark\n"
      << " Sweeping pulse size:\n"
      << "   -1 -- Out-of-place, phased\n"
      << "   -2 -- On-device, phased\n"
      << "   -3 -- On-device, interleaved\n"
      << "\n"
      << " Parameters (for sweeping convolution size, cases 1 through 10)\n"
      << "  -p:rows ROWS -- set number of pulses (default 64)\n"
      << "\n"
      << " Sweeping number of pulses:\n"
      << "  -11 -- Out-of-place, phased\n"
      << "  -12 -- On-device, phased\n"
      << "  -13 -- On-device, interleaved\n"
      << "\n"
      << " Parameters (for sweeping number of convolutions, cases 11 through 20)\n"
      << "  -p:size SIZE -- size of pulse (default 2048)\n"
      ;

  default: return 0;
  }
  return 1;
}

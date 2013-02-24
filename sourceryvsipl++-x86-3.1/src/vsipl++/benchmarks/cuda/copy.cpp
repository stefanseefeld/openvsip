/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for CUDA memory copy.

#include <iostream>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip/opt/cuda/kernels.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;
using namespace vsip_csl;


#define DEBUG 0

// ImplTags
struct Impl_host2dev;
struct Impl_dev2host;
struct Impl_host2host;
struct Impl_dev2shared;
struct Impl_dev2dev;
struct Impl_scalar2dev;

template <typename T,
          typename ImplTag>
struct t_copy_base;


// Helper function to determine the device properties and extract
// the maximum allowed pitch for use with 2-D memory allocation
// and copy functions.
static size_t maximum_pitch = 0;

static
void
initialize_maximum_pitch()
{
  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    printf("There is no device supporting CUDA\n");

  int dev = 0;
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, dev);
  maximum_pitch = device_prop.memPitch;
}





/***********************************************************************
  Vector copy - host to device
***********************************************************************/

template <typename T>
struct t_copy_base<T, Impl_host2dev> : Benchmark_base
{
  typedef Dense<2, T, row2_type>  src_block_t;
  typedef Dense<2, T, row2_type>  dst_block_t;
  
  char const* what() { return "CUDA t_copy_base<..., Impl_host2dev>"; }

  void copy(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T, src_block_t>   A(rows, cols);
    Matrix<T, dst_block_t>   Z(rows, cols, T());
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
        A.put(i, j, T(i * cols + j));

    // Scoping is used to control the lifetime of dda::Data<> objects.  These 
    // must be destroyed before accessing data through the view again.
    {     
      vsip::dda::Data<src_block_t, vsip::dda::in> ext_a(A.block());
      vsip::dda::Data<dst_block_t, vsip::dda::out> ext_z(Z.block());
      T const* pA = ext_a.ptr();
      T* pZ = ext_z.ptr();

      // There is a limit to the row-stride available for use with 2-D
      // memory functions, therefore use the 1-D functions for larger
      // widths (as well as cases where the view is actually a Vector).
      if (rows == 1 || cols == 1 || cols*sizeof(T) > maximum_pitch)
      {
        // Allocate device memory (1-D)
        length_type size = rows * cols;
        T* dev;
        cudaMalloc((void**)&dev, size*sizeof(T));
        
        // Benchmark the operation
        vsip_csl::profile::Timer t1;
        t1.start();
        for (index_type l = 0; l < loop; ++l)
          cudaMemcpy(dev, pA, size*sizeof(T), cudaMemcpyHostToDevice);
        t1.stop();
        time = t1.delta();

        // Pull data back, free device memory.
        cudaMemcpy(pZ, dev, size*sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(dev);
      }
      else
      {
        // Allocate device memory (2-D, using recommended pitch function)
        // The allocated size is rows*pitch
        T* dev;
        size_t pitch = 0;
        cudaMallocPitch((void**)&dev, &pitch, cols*sizeof(T), rows);

        // This is the "pitch" or stride in the host memory (in bytes)
        size_t stride = cols*sizeof(T);
        
        // Benchmark the operation
        vsip_csl::profile::Timer t1;
        t1.start();
        for (index_type l = 0; l < loop; ++l)
          cudaMemcpy2D(dev, pitch, pA, stride, stride, rows, cudaMemcpyHostToDevice);
        t1.stop();
        time = t1.delta();
        
        // Pull data back, free device memory.
        cudaMemcpy2D(pZ, stride, dev, pitch, stride, rows, cudaMemcpyDeviceToHost);
        cudaFree(dev);
      }        
    }

    // validate results
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
      {
#if DEBUG
        if (!equal(A.get(i, j), Z.get(i, j)))
        {
          std::cout << "ERROR: at location " << i << ", " << j << std::endl
                    << "       expected: " << A.get(i, j) << std::endl
                    << "       got     : " << Z.get(i, j) << std::endl;
        }
#endif
        test_assert(equal(A.get(i, j), Z.get(i, j)));
      }

  }
};


/***********************************************************************
  Matrix copy - device to host
***********************************************************************/

template <typename T>
struct t_copy_base<T, Impl_dev2host> : Benchmark_base
{
  typedef Dense<2, T, row2_type>  src_block_t;
  typedef Dense<2, T, row2_type>  dst_block_t;
  
  char const* what() { return "CUDA t_copy_base<..., Impl_dev2host>"; }

  void copy(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T, src_block_t>   A(rows, cols);
    Matrix<T, dst_block_t>   Z(rows, cols, T());
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
        A.put(i, j, T(i * cols + j));

    // Scoping is used to control the lifetime of dda::Data<> objects.  These 
    // must be destroyed before accessing data through the view again.
    {     
      vsip::dda::Data<src_block_t, vsip::dda::in> ext_a(A.block());
      vsip::dda::Data<dst_block_t, vsip::dda::out> ext_z(Z.block());
      T const *pA = ext_a.ptr();
      T* pZ = ext_z.ptr();

      if (rows == 1 || cols == 1 || cols*sizeof(T) > maximum_pitch)
      {
        // Allocate device memory
        length_type size = rows * cols;
        T* dev;
        cudaMalloc((void**)&dev, size*sizeof(T));

        // Push data to device memory
        cudaMemcpy(dev, pA, size*sizeof(T), cudaMemcpyHostToDevice);

        // Benchmark the operation
        vsip_csl::profile::Timer t1;
        t1.start();
        for (index_type l = 0; l < loop; ++l)
          cudaMemcpy(pZ, dev, size*sizeof(T), cudaMemcpyDeviceToHost);
        t1.stop();
        time = t1.delta();

        // Free device memory.
        cudaFree(dev);
      }
      else
      {
        // Allocate device memory (2-D, using recommended pitch function)
        T* dev;
        size_t pitch = 0;
        cudaMallocPitch((void**)&dev, &pitch, cols*sizeof(T), rows);
        
        // This is the "pitch" or stride in the host memory (in bytes)
        size_t stride = cols*sizeof(T);
        
        // Push data to device memory.
        cudaMemcpy2D(dev, pitch, pA, stride, stride, rows, cudaMemcpyHostToDevice);

        // Benchmark the operation
        vsip_csl::profile::Timer t1;
        t1.start();
        for (index_type l = 0; l < loop; ++l)
          cudaMemcpy2D(pZ, stride, dev, pitch, stride, rows, cudaMemcpyDeviceToHost);
        t1.stop();
        time = t1.delta();
        
        cudaFree(dev);
      }        
    }

    // validate results
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
      {
#if DEBUG
        if (!equal(A.get(i, j), Z.get(i, j)))
        {
          std::cout << "ERROR: at location " << i << ", " << j << std::endl
                    << "       expected: " << A.get(i, j) << std::endl
                    << "       got     : " << Z.get(i, j) << std::endl;
        }
#endif
        test_assert(equal(A.get(i, j), Z.get(i, j)));
      }

  }
};


/***********************************************************************
  Matrix copy - host to host
***********************************************************************/

template <typename T>
struct t_copy_base<T, Impl_host2host> : Benchmark_base
{
  typedef Dense<2, T, row2_type>  src_block_t;
  typedef Dense<2, T, row2_type>  dst_block_t;
  
  char const* what() { return "CUDA t_copy_base<..., Impl_host2host>"; }

  void copy(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T, src_block_t>   A(rows, cols);
    Matrix<T, dst_block_t>   Z(rows, cols, T());
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
        A.put(i, j, T(i * cols + j));

    // Scoping is used to control the lifetime of dda::Data<> objects.  These 
    // must be destroyed before accessing data through the view again.
    {     
      vsip::dda::Data<src_block_t, vsip::dda::in> ext_a(A.block());
      vsip::dda::Data<dst_block_t, vsip::dda::out> ext_z(Z.block());
      T const *pA = ext_a.ptr();
      T* pZ = ext_z.ptr();

      if (rows == 1 || cols == 1 || cols*sizeof(T) > maximum_pitch)
      {
        // Allocate device memory
        length_type size = rows * cols;
        T* dev;
        cudaMalloc((void**)&dev, size*sizeof(T));

        // Benchmark the operation
        vsip_csl::profile::Timer t1;
        t1.start();
        for (index_type l = 0; l < loop; ++l)
        {
          cudaMemcpy(dev, pA, size*sizeof(T), cudaMemcpyHostToDevice);
          cudaMemcpy(pZ, dev, size*sizeof(T), cudaMemcpyDeviceToHost);
        }
        t1.stop();
        time = t1.delta();

        // Free device memory.
        cudaFree(dev);
      }
      else
      {
        // Allocate device memory (2-D, using recommended pitch function)
        T* dev;
        size_t pitch = 0;
        cudaMallocPitch((void**)&dev, &pitch, cols*sizeof(T), rows);
        
        // This is the "pitch" or stride in the host memory (in bytes)
        size_t stride = cols*sizeof(T);
        
        // Benchmark the operation
        vsip_csl::profile::Timer t1;
        t1.start();
        for (index_type l = 0; l < loop; ++l)
        {
          cudaMemcpy2D(dev, pitch, pA, stride, stride, rows, cudaMemcpyHostToDevice);
          cudaMemcpy2D(pZ, stride, dev, pitch, stride, rows, cudaMemcpyDeviceToHost);
        }
        t1.stop();
        time = t1.delta();

        // Free device memory.
        cudaFree(dev);
      }
    }

    // validate results
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
      {
#if DEBUG
        if (!equal(A.get(i, j), Z.get(i, j)))
        {
          std::cout << "ERROR: at location " << i << ", " << j << std::endl
                    << "       expected: " << A.get(i, j) << std::endl
                    << "       got     : " << Z.get(i, j) << std::endl;
        }
#endif
        test_assert(equal(A.get(i, j), Z.get(i, j)));
      }

  }
};


/***********************************************************************
  Matrix copy - device to shared (on-chip fast RAM)
***********************************************************************/

template <typename T>
struct t_copy_base<T, Impl_dev2shared> : Benchmark_base
{
  typedef Dense<2, T, row2_type>  src_block_t;
  typedef Dense<2, T, row2_type>  dst_block_t;
  
  char const* what() { return "CUDA t_copy_base<..., Impl_dev2shared>"; }

  void copy(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T, src_block_t>   A(rows, cols);
    Matrix<T, dst_block_t>   Z(rows, cols, T());
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
        A.put(i, j, T(i * cols + j));

    // Scoping is used to control the lifetime of dda::Data<> objects.  These 
    // must be destroyed before accessing data through the view again.
    {     
      vsip::dda::Data<src_block_t, vsip::dda::in> ext_a(A.block());
      vsip::dda::Data<dst_block_t, vsip::dda::out> ext_z(Z.block());
      T const *pA = ext_a.ptr();
      T *pZ = ext_z.ptr();

      // Allocate device memory, copy data from host
      length_type size = rows * cols;
      T* dev;
      cudaMalloc((void**)&dev, size*sizeof(T));
      cudaMemcpy(dev, pA, size*sizeof(T), cudaMemcpyHostToDevice);

    
      // Benchmark the operation
      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l = 0; l < loop; ++l)
        vsip::impl::cuda::copy_device_to_shared(dev, size);
      t1.stop();
      time = t1.delta();


      // Pull data back, free device memory.
      cudaMemcpy(pZ, dev, size*sizeof(T), cudaMemcpyDeviceToHost);
      cudaFree(dev);
    }

    // validate results
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
      {
#if DEBUG
        if (!equal(A.get(i, j), Z.get(i, j)))
        {
          std::cout << "ERROR: at location " << i << ", " << j << std::endl
                    << "       expected: " << A.get(i, j) << std::endl
                    << "       got     : " << Z.get(i, j) << std::endl;
        }
#endif
        test_assert(equal(A.get(i, j), Z.get(i, j)));
      }
    
  }
};


/***********************************************************************
  Matrix copy - device to device
***********************************************************************/

template <typename T>
struct t_copy_base<T, Impl_dev2dev> : Benchmark_base
{
  typedef Dense<2, T, row2_type>  src_block_t;
  typedef Dense<2, T, row2_type>  dst_block_t;
  
  char const* what() { return "CUDA t_copy_base<..., Impl_dev2dev>"; }

  void copy(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T, src_block_t>   A(rows, cols);
    Matrix<T, dst_block_t>   Z(rows, cols, T());
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
        A.put(i, j, T(i * cols + j));

    // Scoping is used to control the lifetime of dda::Data<> objects.  These 
    // must be destroyed before accessing data through the view again.
    {     
      vsip::dda::Data<src_block_t, vsip::dda::in> ext_a(A.block());
      vsip::dda::Data<dst_block_t, vsip::dda::out> ext_z(Z.block());
      T const *pA = ext_a.ptr();
      T *pZ = ext_z.ptr();

      // Allocate device memory, copy data from host
      length_type size = rows * cols;
      T* dev_a;
      cudaMalloc((void**)&dev_a, size*sizeof(T));
      cudaMemcpy(dev_a, pA, size*sizeof(T), cudaMemcpyHostToDevice);
      T* dev_z;
      cudaMalloc((void**)&dev_z, size*sizeof(T));

    
      // Benchmark the operation
      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l = 0; l < loop; ++l)
      {
        vsip::impl::cuda::copy(dev_a, dev_z, size);
        cudaThreadSynchronize();
      }
      t1.stop();
      time = t1.delta();


      // Pull data back, free device memory.
      cudaMemcpy(pZ, dev_z, size*sizeof(T), cudaMemcpyDeviceToHost);
      cudaFree(dev_z);
      cudaFree(dev_a);
    }

    // validate results
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
      {
#if DEBUG
        if (!equal(A.get(i, j), Z.get(i, j)))
        {
          std::cout << "ERROR: at location " << i << ", " << j << std::endl
                    << "       expected: " << A.get(i, j) << std::endl
                    << "       got     : " << Z.get(i, j) << std::endl;
        }
#endif
        test_assert(equal(A.get(i, j), Z.get(i, j)));
      }
    
  }
};


/***********************************************************************
  Matrix copy - device fill with scalar
***********************************************************************/

template <typename T>
struct t_copy_base<T, Impl_scalar2dev> : Benchmark_base
{
  typedef Dense<2, T, row2_type>  src_block_t;
  typedef Dense<2, T, row2_type>  dst_block_t;
  
  char const* what() { return "CUDA t_copy_base<..., Impl_scalar2dev>"; }

  void copy(length_type rows, length_type cols, length_type loop, float& time)
  {
    Matrix<T, src_block_t>   A(rows, cols);
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
        A.put(i, j, T(i * cols + j));

    // Scoping is used to control the lifetime of dda::Data<> objects.  These 
    // must be destroyed before accessing data through the view again.
    {     
      vsip::dda::Data<src_block_t, vsip::dda::out> ext_a(A.block());
      T* pA = ext_a.ptr();

      // Allocate device memory, copy data from host
      length_type size = rows * cols;
      T* dev_a;
      cudaMalloc((void**)&dev_a, size*sizeof(T));
      cudaMemcpy(dev_a, pA, size*sizeof(T), cudaMemcpyHostToDevice);

    
      // Benchmark the operation
      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l = 0; l < loop; ++l)
      {
        vsip::impl::cuda::assign_scalar(T(0), dev_a, size);
        cudaThreadSynchronize();
      }
      t1.stop();
      time = t1.delta();


      // Pull data back, free device memory.
      cudaMemcpy(pA, dev_a, size*sizeof(T), cudaMemcpyDeviceToHost);
      cudaFree(dev_a);
    }

    // validate results
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
      {
#if DEBUG
        if (!equal(A.get(i, j), T()))
        {
          std::cout << "ERROR: at location " << i << ", " << j << std::endl
                    << "       expected: " << T() << std::endl
                    << "       got     : " << A.get(i, j) << std::endl;
        }
#endif
        test_assert(equal(A.get(i, j), T()));
      }

  }
};


/***********************************************************************
  Row size constant
***********************************************************************/

template <typename T, typename ImplTag>
struct t_copy_rows_fixed : public t_copy_base<T, ImplTag>
{
  int ops_per_point(length_type)  { return rows_; }
  int riob_per_point(length_type) { return rows_*sizeof(T); }
  int wiob_per_point(length_type) { return rows_*sizeof(T); }
  int mem_per_point(length_type)  { return rows_*sizeof(T); }

  void operator()(vsip::length_type size, vsip::length_type loop, float& time)
  {
    this->copy(rows_, size, loop, time);
  }

  t_copy_rows_fixed(vsip::length_type rows) : rows_(rows) {}

// Member data
  vsip::length_type rows_;
};


/***********************************************************************
  Column size constant
***********************************************************************/

template <typename T, typename ImplTag>
struct t_copy_cols_fixed : public t_copy_base<T, ImplTag>
{
  int ops_per_point(length_type)  { return cols_; }
  int riob_per_point(length_type) { return cols_*sizeof(T); }
  int wiob_per_point(length_type) { return cols_*sizeof(T); }
  int mem_per_point(length_type)  { return cols_*sizeof(T); }

  void operator()(vsip::length_type size, vsip::length_type loop, float& time)
  {
    this->copy(size, cols_, loop, time);
  }

  t_copy_cols_fixed(vsip::length_type cols) : cols_(cols) {}

// Member data
  vsip::length_type cols_;
};



/***********************************************************************
  Benchmark Driver
***********************************************************************/

void
defaults(Loop1P& loop)
{
  loop.param_["rows"] = "64";
  loop.param_["size"] = "2048";
}



int
test(Loop1P& loop, int what)
{
  typedef float F;
  initialize_maximum_pitch();

  length_type rows  = atoi(loop.param_["rows"].c_str());
  length_type size  = atoi(loop.param_["size"].c_str());

  std::cout << "rows: " << rows << "  size: " << size 
	    << std::endl;


  switch (what)
  {
  case  1: loop(t_copy_rows_fixed<F, Impl_host2dev>(rows)); break;
  case  2: loop(t_copy_rows_fixed<F, Impl_dev2host>(rows)); break;
  case  3: loop(t_copy_rows_fixed<F, Impl_host2host>(rows)); break;
  case  4: loop(t_copy_rows_fixed<F, Impl_dev2shared>(rows)); break;
  case  5: loop(t_copy_rows_fixed<F, Impl_dev2dev>(rows)); break;
  case  6: loop(t_copy_rows_fixed<F, Impl_scalar2dev>(rows)); break;

  case 11: loop(t_copy_cols_fixed<F, Impl_host2dev>(size)); break;
  case 12: loop(t_copy_cols_fixed<F, Impl_dev2host>(size)); break;
  case 13: loop(t_copy_cols_fixed<F, Impl_host2host>(size)); break;
  case 14: loop(t_copy_cols_fixed<F, Impl_dev2shared>(size)); break;
  case 15: loop(t_copy_cols_fixed<F, Impl_dev2dev>(size)); break;
  case 16: loop(t_copy_cols_fixed<F, Impl_scalar2dev>(size)); break;

  case 0:
    std::cout
      << "CUDA copy -- fixed rows\n"
      << "   -1 -- host to device copy\n"
      << "   -2 -- device to host copy\n"
      << "   -3 -- host->device->host copy (A = B)\n"
      << "   -4 -- device to shared copy\n"
      << "   -5 -- device to device copy\n"
      << "   -6 -- device fill with scalar\n"
      ;

    std::cout
      << "CUDA copy -- fixed columns\n"
      << "  -11 -- host to device copy\n"
      << "  -12 -- device to host copy\n"
      << "  -13 -- host->device->host copy (A = B)\n"
      << "  -14 -- device to shared copy\n"
      << "  -15 -- device to device copy\n"
      << "  -16 -- device fill with scalar\n"
      ;

  default:
    return 0;
  }
  return 1;
}

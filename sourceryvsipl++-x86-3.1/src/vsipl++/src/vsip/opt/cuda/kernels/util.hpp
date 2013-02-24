/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    util.hpp
    @author  Don McCoy
    @date    2009-06-09
    @brief   VSIPL++ Library: Utility functions for CUDA Kernels
*/

#ifndef VSIP_OPT_CUDA_KERNELS_UTIL_HPP
#define VSIP_OPT_CUDA_KERNELS_UTIL_HPP


/***********************************************************************
  Included Files
***********************************************************************/

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <vector_types.h>


/***********************************************************************
  Definitions
***********************************************************************/

/// Technical specifications and features are determined by a device's
/// compute capability.  This information was taken from NVIDIA CUDA
/// Programming Guide 2.2, Appendix A - Technical Specifications.
enum compute_capability
{
  CC_1_0,
  CC_1_1,
  CC_1_2,
  CC_1_3
};

/// Baseline CUDA properties, Compute Capability 1.0
template <enum compute_capability CC = CC_1_0>
struct Device_properties
{
  static size_t shared_memory_size()              { return 16384; }
  static size_t constant_memory_size()            { return 65536; }
  static size_t registers_per_block()             { return 8192; }
  static size_t warp_size()                       { return 32; }
  // for 1-D views:
  static size_t max_threads_per_block()           { return 512; }
  // for 2-D views the above is divided as evenly as possible
  static size_t max_threads_per_block_x()         { return 32; }
  static size_t max_threads_per_block_y()         { return 16; }
  static size_t max_threads_per_dimension(int d)  { return (d == 0 ? 512 : 
                                                           (d == 1 ? 512 : 
                                                           (d == 2 ? 64 : 
                                                             0 ))); }
  static size_t max_grid_dimension(int d)         { return (d == 0 ? 65535 : 
                                                           (d == 1 ? 65535 : 
                                                           (d == 2 ? 1 : 
                                                             0 ))); }
  static bool   supports_32bit_atomic()           { return false; }
  static bool   supports_64bit_shared_atomic()    { return false; }
  static bool   supports_double()                 { return false; }
};

/// Specialization for Compute Capability 1.1
template <>
struct Device_properties<CC_1_1> : Device_properties<CC_1_0>
{
  static bool   supports_32bit_atomic()           { return true; }
};

/// Specialization for Compute Capability 1.2
template <>
struct Device_properties<CC_1_2> : Device_properties<CC_1_1>
{
  static size_t registers_per_block()             { return 16384; }
  static bool   supports_64bit_shared_atomic()    { return true; }
};

/// Specialization for Compute Capability 1.3
template <>
struct Device_properties<CC_1_3> : Device_properties<CC_1_2>
{
  static bool   supports_double()                 { return true; }
};


/// The above static members should be accessed through this class
/// in order to default to a particular compute capabilty.  
///
/// In this case, 1.1 is a good choice for a default because most
/// hardware meets this as a minimum.
struct Dev_props : Device_properties<CC_1_1>
{};




/***********************************************************************
  Helper functions
***********************************************************************/

/// Calculates the optimal grid and thread-block sizes for a simple
/// distribution of a vector.  At least one thread for every element 
/// is created, with the actual number rounded up in each dimension
/// to create fully populated blocks.  Excess threads must be kept
/// idle by ensuring they are within bounds of the actual vector.
///
///   :grid:  set to the minimum number of thread blocks
///           needed to accomodate a matrix of the given size
///
///   :threads:  set to the size of the thread block
///
inline 
void
distribute_vector(size_t size, dim3& grid, dim3& threads)
{
  threads.x = Dev_props::max_threads_per_block();
  grid.x = (size + threads.x - 1) / threads.x;
}


/// Calculates the optimal grid and thread-block sizes for a simple
/// distribution of a matrix.  At least one thread for every element 
/// is created, with the actual number rounded up in each dimension
/// to create fully populated blocks.  Excess threads must be kept
/// idle by ensuring they are within bounds of the actual matrix.
///
///   :grid:  set to the minimum number of thread blocks
///           needed to accomodate a matrix of the given size
///
///   :threads:  set to the size of the thread block
///
inline 
void
distribute_matrix(size_t rows, size_t cols, dim3& grid, dim3& threads)
{
  // Optimal thread blocks are as large as possible in order to 
  // maximize occupancy.  Choosing a constant division results in
  // a performance penalty for small matrices because some threads
  // will be idle (those outside the bounds of the actual matrix).
  // 
  threads.y = Dev_props::max_threads_per_block_y();
  threads.x = Dev_props::max_threads_per_block_x();

  grid.y = (rows + threads.y - 1) / threads.y;
  grid.x = (cols + threads.x - 1) / threads.x;
}


/// Checks to see if a given view exceeds the maximum size allowed,
/// meaning it cannot be distributed with a single kernel invocation.
/// Whether or not it can be broken down into smaller problem size
/// is algorithm-dependent.
inline
bool
vector_is_distributable(size_t size)
{
  dim3 grid;
  dim3 threads;

  distribute_vector(size, grid, threads);

  return 
    (threads.x <= Dev_props::max_threads_per_dimension(0)) &&
    (grid.x <= Dev_props::max_grid_dimension(0));
}


/// Checks to see if a given view exceeds the maximum size allowed,
/// meaning it cannot be distributed with a single kernel invocation.
/// Whether or not it can be broken down into smaller problem size
/// is algorithm-dependent.
inline
bool
matrix_is_distributable(size_t rows, size_t cols)
{
  dim3 grid;
  dim3 threads;

  distribute_matrix(rows, cols, grid, threads);

  return 
    (threads.y <= Dev_props::max_threads_per_dimension(1)) &&
    (threads.x <= Dev_props::max_threads_per_dimension(0)) &&
    (grid.y <= Dev_props::max_grid_dimension(1)) &&
    (grid.x <= Dev_props::max_grid_dimension(0));
}



#endif // VSIP_OPT_CUDA_KERNELS_UTIL_HPP

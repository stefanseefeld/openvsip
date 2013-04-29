/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Test data transfer between host and device memory.
///   On the device we use an external (and trusted) copy function.
///   We validate data on the host.
///   Thus, we can at least be sure that the host->dev and dev->host
///   transfers do the same.

#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/selgen.hpp>
#include <vsip/random.hpp>
#include <vsip/opt/cuda/stored_block.hpp>
#include <vsip_csl/test.hpp>

#ifndef DEBUG
#define DEBUG 1
#endif

#if DEBUG
#include <vsip_csl/output.hpp>
#endif

using namespace vsip;
namespace cuda = vsip::impl::cuda;

/// On the device, split-complex data occupy contiguous memory,
/// so a copy operation may just copy the bytes in a single call,
/// Using the first pointer.
template <typename T>
T *convert_ptr(T *ptr) { return ptr;}
template <typename T>
T *convert_ptr(std::pair<T*,T*> ptr) { return ptr.first;}

template <typename T, storage_format_type C>
void copy_test(length_type N)
{
  typedef Layout<1, tuple<0,1,2>, dense, C> layout_type;
  typedef cuda::Stored_block<T, layout_type> block_type;
  typedef Vector<T, block_type> view_type;
  view_type a = ramp(1., 2., N);
  view_type b(N);
  cudaMemcpy(convert_ptr(b.block().device_ptr()),
	     convert_ptr(a.block().device_ptr()),
	     N * sizeof(T), 
	     cudaMemcpyDeviceToDevice);
  test_assert(vsip_csl::view_equal(a, b));
}

template <typename T, typename O, storage_format_type C>
void copy_test_dense_2d(length_type N, length_type M)
{
  typedef Layout<2, O, dense, C> layout_type;
  typedef cuda::Stored_block<T, layout_type> block_type;
  typedef Matrix<T, block_type> view_type;
  view_type a = Rand<T>(0).randu(N, M);
  view_type b(N, M);
  cudaMemcpy(convert_ptr(b.block().device_ptr()),
	     convert_ptr(a.block().device_ptr()),
	     N * M * sizeof(T), 
	     cudaMemcpyDeviceToDevice);
  test_assert(vsip_csl::view_equal(a, b));
}

template <typename T, typename O, storage_format_type C>
void copy_test_aligned_2d(length_type N, length_type M)
{
  typedef Layout<2, O, aligned_128, C> layout_type;
  typedef cuda::Stored_block<T, layout_type> block_type;
  typedef Matrix<T, block_type> view_type;
  view_type a = Rand<T>(0).randu(N, M);
  view_type b(N, M);
  cudaMemcpy(convert_ptr(b.block().device_ptr()),
	     convert_ptr(a.block().device_ptr()),
	     N * M * sizeof(T), 
	     cudaMemcpyDeviceToDevice);
  test_assert(vsip_csl::view_equal(a, b));
}


int
main(int argc, char** argv)
{
  vsipl library(argc, argv);
  copy_test<float, interleaved_complex>(256);
  copy_test<complex<float>, interleaved_complex>(256);
  copy_test<complex<float>, split_complex>(256);

  copy_test_dense_2d<float, tuple<0,1,2>, interleaved_complex>(257, 259);
  copy_test_dense_2d<complex<float>, tuple<0,1,2>, interleaved_complex>(257, 259);
  copy_test_dense_2d<complex<float>, tuple<0,1,2>, split_complex>(257, 259);
  copy_test_dense_2d<float, tuple<1,0,2>, interleaved_complex>(257, 259);
  copy_test_dense_2d<complex<float>, tuple<1,0,2>, interleaved_complex>(257, 259);
  copy_test_dense_2d<complex<float>, tuple<1,0,2>, split_complex>(257, 259);

  copy_test_aligned_2d<float, tuple<0,1,2>, interleaved_complex>(257, 259);
  copy_test_aligned_2d<complex<float>, tuple<0,1,2>, interleaved_complex>(257, 259);
  copy_test_aligned_2d<complex<float>, tuple<0,1,2>, split_complex>(257, 259);
  copy_test_aligned_2d<float, tuple<1,0,2>, interleaved_complex>(257, 259);
  copy_test_aligned_2d<complex<float>, tuple<1,0,2>, interleaved_complex>(257, 259);
  copy_test_aligned_2d<complex<float>, tuple<1,0,2>, split_complex>(257, 259);
}

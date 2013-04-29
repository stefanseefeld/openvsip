/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Test runtime direct data access, in particular those involving 
///   dimension reorder (transpose) as well as interleaved <-> split 
///   changes (shuffles).
///   We construct two blocks with a given layout, but then use DDA with a
///   different layout, forcing device-side copies to be made.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/selgen.hpp>
#include <vsip/random.hpp>
#include <vsip/opt/cuda/stored_block.hpp>
#include <vsip/opt/cuda/dda.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/diagnostics.hpp>
#include <iostream>


#ifndef DEBUG
#define DEBUG 1
#endif

#if DEBUG
#include <vsip_csl/output.hpp>
#endif

using namespace vsip;

template <typename L1, typename L2>
void copy_test(length_type N)
{
#ifdef DEBUG
  std::cout << "testing " 
	    << L1::storage_format << ' ' 
	    << " <-> "
	    << L2::storage_format << std::endl;
#endif
  using namespace vsip::impl;
  typedef complex<float> C;
  typedef Strided<1, C, L1> block_type;
  typedef Vector<C, block_type> view_type;
  view_type a = ramp(C(0),C(1,2), N);
  view_type b(N, 0);
  Rt_layout<1> rtl(dense, Rt_tuple(tuple<0,1,2>()), L2::storage_format, 0);
  {
    cuda::dda::Rt_data<block_type, dda::in> a_ptr(a.block(), rtl);
    cuda::dda::Rt_data<block_type, dda::out> b_ptr(b.block(), rtl);

    cudaMemcpy(b_ptr.ptr().as_inter(),
	       a_ptr.ptr().as_inter(),
	       N * sizeof(C), 
	       cudaMemcpyDeviceToDevice);
  }
#ifdef DEBUG
  if (!vsip_csl::view_equal(a, b))
  {
    std::cout << "a:\n" << a << std::endl;
    std::cout << "b:\n" << b << std::endl;
  }
#endif
  test_assert(vsip_csl::view_equal(a, b));
}

template <typename L1, typename L2>
void copy_test(length_type N, length_type M)
{
#ifdef DEBUG
  std::cout << "testing " 
	    << vsip_csl::type_name<typename L1::order_type>() << ' ' 
	    << L1::storage_format << ' '
	    << " <-> "
	    << vsip_csl::type_name<typename L2::order_type>() << ' '
	    << L2::storage_format << std::endl;
#endif
  using namespace vsip::impl;
  typedef complex<float> C;
  typedef Strided<2, C, L1> block_type;
  typedef Matrix<C, block_type> view_type;
  view_type a(N, M);
  for (index_type i = 0; i != N; ++i)
    a.row(i) = ramp(C(i,2*i), C(N,2*N), M);
  view_type b(N, M, 0);
  Rt_layout<2> rtl(dense, Rt_tuple(typename L2::order_type()), L2::storage_format, 0);
  {
    cuda::dda::Rt_data<block_type, dda::in> a_ptr(a.block(), rtl);
    cuda::dda::Rt_data<block_type, dda::out> b_ptr(b.block(), rtl);

    cudaMemcpy(b_ptr.ptr().as_inter(),
	       a_ptr.ptr().as_inter(),
	       N * M * sizeof(C), 
	       cudaMemcpyDeviceToDevice);
  }
#ifdef DEBUG
  if (!vsip_csl::view_equal(a, b))
  {
    std::cout << "a:\n" << a << std::endl;
    std::cout << "b:\n" << b << std::endl;
  }
#endif
  test_assert(vsip_csl::view_equal(a, b));
}

int
main(int argc, char** argv)
{
  using namespace vsip::impl;
  typedef Layout<1, tuple<0,1,2>, dense, interleaved_complex> inter;
  typedef Layout<1, tuple<0,1,2>, dense, split_complex> split;
  typedef Layout<2, tuple<0,1,2>, dense, interleaved_complex> row_inter;
  typedef Layout<2, tuple<0,1,2>, dense, split_complex> row_split;
  typedef Layout<2, tuple<1,0,2>, dense, interleaved_complex> col_inter;
  typedef Layout<2, tuple<1,0,2>, dense, split_complex> col_split;

  vsipl library(argc, argv);

  // 1D copies
  copy_test<inter, inter>(16);
  copy_test<inter, split>(16);
  copy_test<split, inter>(16);
  copy_test<split, split>(16);

  // 2D copies
  copy_test<row_inter, row_inter>(16, 8);
  copy_test<row_inter, row_split>(16, 8);
  copy_test<row_inter, col_inter>(16, 8);
  copy_test<row_inter, col_split>(16, 8);
  copy_test<row_split, row_inter>(16, 8);
  copy_test<row_split, row_split>(16, 8);
  copy_test<row_split, col_inter>(16, 8);
  copy_test<row_split, col_split>(16, 8);
  copy_test<col_inter, row_inter>(16, 8);
  copy_test<col_inter, row_split>(16, 8);
  copy_test<col_inter, col_inter>(16, 8);
  copy_test<col_inter, col_split>(16, 8);
  copy_test<col_split, row_inter>(16, 8);
  copy_test<col_split, row_split>(16, 8);
  copy_test<col_split, col_inter>(16, 8);
  copy_test<col_split, col_split>(16, 8);
  return 0;
}

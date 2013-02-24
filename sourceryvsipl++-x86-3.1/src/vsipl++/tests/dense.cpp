/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/dense.cpp
    @author  Jules Bergmann
    @date    2005-01-24
    @brief   VSIPL++ Library: Unit tests for Dense blocks.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/dense.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;

/***********************************************************************
  Definitions
***********************************************************************/

template <typename Order>
index_type
linear_index(index_type idx0, index_type idx1,
	      length_type size0, length_type size1);

template <>
index_type
linear_index<row2_type>(index_type idx0, index_type idx1,
			 length_type /*size0*/, length_type size1)
{
  return size1*idx0+idx1;
}

template <>
index_type
linear_index<col2_type>(index_type idx0, index_type idx1,
			 length_type size0, length_type /*size1*/)
{
  return idx0+size0*idx1;
}



/// Check that a block's 1-dimensional and 2-dimensional accessors
/// are consistent with a data layout.
template <typename T,
	  typename Order>
void
check_order(Dense<2, T, Order>& block)
{
  for (index_type i=0; i<block.size(2, 0); ++i)
    for (index_type j=0; j<block.size(2, 1); ++j)
      block.put(i, j, T(100*i + j));

  for (index_type i=0; i<block.size(2, 0); ++i)
  {
    for (index_type j=0; j<block.size(2, 1); ++j)
    {
      index_type idx = linear_index<Order>(i, j, block.size(2, 0), block.size(2, 1));
      test_assert(equal(block.get(idx), T(100*i + j)));
      block.put(idx, T(i + 1000*j));
    }
  }

  for (index_type i=0; i<block.size(2, 0); ++i)
    for (index_type j=0; j<block.size(2, 1); ++j)
      test_assert(equal(block.get(i, j), T(i + 1000*j)));
}



/// Checks that block can be passed as const arguement.
template <typename T>
void
check_block_const(Dense<1, T> const& block)
{
  // Increment the block reference count to make sure that the block
  // handles reference counts as a mutable value.

  block.increment_count();

  for (index_type i=0; i<block.size(); ++i)
    test_assert(equal(block.get(i), T(2*i)));

  block.decrement_count();
}



/// Simple check of a 1-dimensional block's get() and put() accessors.
template <typename T>
void
check_block_gp(Dense<1, T>& block)
{
  for (index_type i=0; i<block.size(); ++i)
    block.put(i, T(2*i));

  for (index_type i=0; i<block.size(); ++i)
    test_assert(equal(block.get(i), T(2*i)));

  check_block_const(block);
}



/// Check 2-dimensional block's get() and put() accessors, and dimension
/// order.
template <typename T,
	  typename Order>
void
check_block_gp(Dense<2, T, Order>& block)
{
  for (index_type i=0; i<block.size(2, 0); ++i)
    for (index_type j=0; j<block.size(2, 1); ++j)
      block.put(i, j, T(100*i + j));

  for (index_type i=0; i<block.size(2, 0); ++i)
    for (index_type j=0; j<block.size(2, 1); ++j)
      test_assert(equal(block.get(i, j), T(100*i + j)));

  check_order(block);
}



/// Check 3-dimensional block's get() and put() accessors, and dimension
/// order.
template <typename T,
	  typename Order>
void
check_block_gp(Dense<3, T, Order>& block)
{
  for (index_type i=0; i<block.size(3, 0); ++i)
    for (index_type j=0; j<block.size(3, 1); ++j)
      for (index_type k=0; k<block.size(3, 2); ++k)
	block.put(i, j, k, T(1000*i + 100*j + k));

  for (index_type i=0; i<block.size(3, 0); ++i)
    for (index_type j=0; j<block.size(3, 1); ++j)
      for (index_type k=0; k<block.size(3, 2); ++k)
      test_assert(equal(block.get(i, j, k),  T(1000*i + 100*j + k)));

  // check_order(block);
}



/// Simple check of a 1-dimensional block's impl_ref() accessor.
template <typename T>
void
check_block_at(Dense<1, T>& block)
{
  // impl_ref() is only valid if block stores complex in interleaved
  // format.  Otherwise lvalue_proxy's are used.

  if (!vsip::impl::is_split_block<Dense<1, T> >::value)
  {
    for (index_type i=0; i<block.size(); ++i)
      block.impl_ref(i) = T(2*i);

    for (index_type i=0; i<block.size(); ++i)
      test_assert(equal(block.impl_ref(i), T(2*i)));
  }
}



/// Simple check of a 2-dimensional block's impl_ref() accessor.
template <typename T,
	  typename Order>
void
check_block_at(Dense<2, T, Order>& block)
{
  if (!vsip::impl::is_split_block<Dense<2, T> >::value)
  {
    for (index_type i=0; i<block.size(2, 0); ++i)
      for (index_type j=0; j<block.size(2, 1); ++j)
	block.impl_ref(i, j) = T(100*i + j);
    
    for (index_type i=0; i<block.size(2, 0); ++i)
      for (index_type j=0; j<block.size(2, 1); ++j)
	test_assert(equal(block.impl_ref(i, j), T(100*i + j)));
  }
}



/// Simple check of a 2-dimensional block's impl_ref() accessor.
template <typename T,
	  typename Order>
void
check_block_at(Dense<3, T, Order>& block)
{
  if (!vsip::impl::is_split_block<Dense<3, T> >::value)
  {
    for (index_type i=0; i<block.size(3, 0); ++i)
      for (index_type j=0; j<block.size(3, 1); ++j)
	for (index_type k=0; k<block.size(3, 2); ++k)
	  block.impl_ref(i, j, k) = T(1000*i + 100*j + k);
    
    for (index_type i=0; i<block.size(3, 0); ++i)
      for (index_type j=0; j<block.size(3, 1); ++j)
	for (index_type k=0; k<block.size(3, 2); ++k)
	  test_assert(equal(block.impl_ref(i, j, k), T(1000*i + 100*j + k)));
  }
}



/// Create Dense block on stack and check functionality.
template <dimension_type Dim,
	  typename       T>
void
test_stack_dense(Domain<Dim> const& dom)
{
  Dense<Dim, T>	block(dom);

  // Initial reference count is 1.  This block will be freed when it
  // goes out of scope.

  // Check out user-storage functions

  test_assert(block.admitted()     == true);
  test_assert(block.user_storage() == no_user_format);

  T* ptr;
  block.find(ptr);
  test_assert(ptr == NULL);

  // Check that block dimension sizes match domain.
  length_type total_size = 1;
  for (dimension_type d=0; d<Dim; ++d)
  {
    test_assert(block.size(Dim, d) == dom[d].size());
    total_size *= block.size(Dim, d);
  }

  test_assert(total_size == block.size());

  check_block_gp(block);
  check_block_at(block);
}



/// Create Dense block on heap and check functionality.
template <dimension_type Dim,
	  typename    T,
	  typename    Order>
void
test_heap_dense(Domain<Dim> const& dom)
{
  Dense<Dim, T, Order>* block = new Dense<Dim, T, Order>(dom);

  // Initial reference count is 1.

  // Check out user-storage functions

  test_assert(block->admitted()     == true);
  test_assert(block->user_storage() == no_user_format);

  T* ptr;
  block->find(ptr);
  test_assert(ptr == NULL);

  // Check that block dimension sizes match domain.
  length_type total_size = 1;
  for (dimension_type d=0; d<Dim; ++d)
  {
    test_assert(block->size(Dim, d) == dom[d].size());
    total_size *= block->size(Dim, d);
  }

  test_assert(total_size == block->size());

  check_block_gp(*block);
  check_block_at(*block);

  block->decrement_count();
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);
  test_stack_dense<1, int>            (Domain<1>(10));
  test_stack_dense<1, float>          (Domain<1>(10));
  test_stack_dense<1, complex<float> >(Domain<1>(10));

  test_stack_dense<2, int>            (Domain<2>(15, 10));
  test_stack_dense<2, float>          (Domain<2>(10, 15));
  test_stack_dense<2, complex<float> >(Domain<2>(10, 15));

  test_stack_dense<3, int>             (Domain<3>(15, 10, 25));
  test_stack_dense<3, float>           (Domain<3>(10, 15, 25));
  test_stack_dense<3, complex<double> >(Domain<3>(10, 15, 25));

  test_heap_dense<1, int,            row1_type>(Domain<1>(10));
  test_heap_dense<1, float,          row1_type>(Domain<1>(10));
  test_heap_dense<1, complex<float>, row1_type>(Domain<1>(10));

  test_heap_dense<2, int,             row2_type>(Domain<2>(10, 10));
  test_heap_dense<2, float,           row2_type>(Domain<2>(10, 15));
  test_heap_dense<2, complex<double>, row2_type>(Domain<2>(16, 8));
  test_heap_dense<2, int,             col2_type>(Domain<2>(10, 10));
  test_heap_dense<2, float,           col2_type>(Domain<2>(10, 15));
  test_heap_dense<2, complex<double>, col2_type>(Domain<2>(15,  5));

  test_heap_dense<3, float,          row3_type>(Domain<3>(15, 5, 3));
  test_heap_dense<3, float,          col3_type>(Domain<3>(5, 7, 15));
  test_heap_dense<3, complex<float>, row3_type>(Domain<3>(5, 7, 3));
  test_heap_dense<3, complex<float>, col3_type>(Domain<3>(3, 5, 7));
}
